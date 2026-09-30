"""
Synthetic dispute generator with known ground truth at every stage.

Each case is built from latent facts -> rendered evidence documents -> outcome:

  1. Facts (yes/no/unknown per fact) are sampled per case. A latent
     "merchant is in the right" variable correlates them, as in real disputes.
  2. Facts are rendered into free-text documents with varied phrasing,
     including negations, name mismatches and numeric comparisons, so that
     reading the documents is a real extraction task rather than a lookup.
  3. The outcome is drawn from a logistic model of the facts *as evidenced
     in the documents* (what the merchant can actually submit) plus the
     merchant's track record. Issuer discretion is the Bernoulli noise.

Because true facts and the true P(win) are saved, the eval can separate
extraction error from model error from irreducible outcome noise.

Splits: train/val/test use the "dev" phrasing bank. test_shifted renders the
same kinds of facts with a held-out phrasing bank written after the rule-based
extractor was frozen, to measure robustness to wording nobody tuned against.

Usage: python scripts/generate_dataset.py [--seed 7] [--n-train 1000 --n-val 250 --n-test 250]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FIRST = ["Priya", "Rahul", "Ananya", "Vikram", "Sneha", "Arjun", "Kavya", "Rohan", "Meera", "Aditya",
         "Divya", "Karthik", "Isha", "Nikhil", "Pooja", "Siddharth"]
LAST = ["Nair", "Sharma", "Iyer", "Menon", "Reddy", "Gupta", "Kulkarni", "Das", "Bose", "Patel",
        "Rao", "Joshi", "Verma", "Pillai"]
SIMILAR_LAST = {"Nair": "Nayak", "Sharma": "Sarma", "Iyer": "Iyengar", "Menon": "Mehta", "Reddy": "Reddi",
                "Gupta": "Gupte", "Kulkarni": "Karnik", "Das": "Dash", "Bose": "Basu", "Patel": "Patil",
                "Rao": "Raju", "Joshi": "Joshua", "Verma": "Varma", "Pillai": "Pillay"}
CITIES = ["Pune", "Chennai", "Bengaluru", "Hyderabad", "Kochi", "Jaipur", "Lucknow", "Kolkata", "Mumbai", "Delhi"]
FAR = ["Lagos, NG", "Bucharest, RO", "Hanoi, VN", "Sao Paulo, BR", "Minsk, BY"]

# True outcome model: per-category intercept and (yes_coef, no_coef) per relevant fact.
TRUE_MODEL = {
    "not_received": (-1.0, {"delivery_confirmed": (1.6, -1.0), "signed_by_cardholder": (1.0, -0.8),
                            "customer_acknowledged_receipt": (2.2, -0.6)}),
    "fraud": (-1.6, {"avs_cvv_match": (0.9, -0.9), "ip_consistent_with_cardholder": (1.0, -1.0),
                     "prior_undisputed_orders": (1.4, -0.7), "delivery_confirmed": (0.4, -0.3),
                     "signed_by_cardholder": (0.8, -0.4)}),
    "not_as_described": (-1.2, {"item_matches_description": (1.5, -1.4), "return_or_refund_offered": (1.2, -0.6)}),
    "incorrect_amount": (-0.6, {"amount_matches_agreement": (2.0, -2.0), "signed_by_cardholder": (0.6, -0.3)}),
}
MERCHANT_EFFECT = 1.5  # logit shift per unit of (historical_win_rate - 0.5)
# Planted process problems for the "portfolio" split only, so the merchant/portfolio
# diagnosis can be checked against known ground truth (scripts/portfolio_demo.py).
PLANTED_MERCHANT_GAPS = {"mch_04": "signed_by_cardholder", "mch_06": "return_or_refund_offered"}
PLANTED_SYSTEMIC_GAP = "prior_undisputed_orders"   # every merchant's pipeline drops account history
PLANTED_RATE = 0.8

RC_WEIGHTS = {"13.1": 3, "C08": 1, "10.4": 3, "4837": 2, "F29": 1, "13.3": 2, "4853": 1, "12.5": 2}
FACTS = ["delivery_confirmed", "signed_by_cardholder", "customer_acknowledged_receipt", "item_matches_description",
         "return_or_refund_offered", "avs_cvv_match", "ip_consistent_with_cardholder", "prior_undisputed_orders",
         "amount_matches_agreement"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Gen:
    def __init__(self, seed: int, style: str = "dev", planted: bool = False):
        self.r = random.Random(seed)
        self.planted = planted
        self.style = style  # "dev" phrasings (rules were written against these) or "shifted" (held out)
        self.reason_codes = {r["code"]: r for r in json.loads((DATA / "reason_codes.json").read_text())}
        self.merchants = json.loads((DATA / "merchants.json").read_text())

    # ---------- facts ----------
    def sample_facts(self, relevant: list[str], category: str) -> dict[str, str]:
        r = self.r
        merchant_right = r.random() < 0.5
        p_yes = 0.8 if merchant_right else 0.3
        facts = {}
        for f in FACTS:
            p_unknown = 0.2 if f in relevant else 0.7
            if r.random() < p_unknown:
                facts[f] = "unknown"
            else:
                facts[f] = "yes" if r.random() < p_yes else "no"
        # A signature only exists where something was handed over.
        if category != "incorrect_amount" and facts["delivery_confirmed"] != "yes":
            facts["signed_by_cardholder"] = "unknown"
        return facts

    def true_p_win(self, category: str, facts: dict[str, str], merchant: dict) -> float:
        intercept, coefs = TRUE_MODEL[category]
        z = intercept + MERCHANT_EFFECT * (merchant["historical_win_rate"] - 0.5)
        for f, (yes_c, no_c) in coefs.items():
            z += yes_c if facts[f] == "yes" else no_c if facts[f] == "no" else 0.0
        return sigmoid(z)

    def plant(self, facts: dict[str, str], merchant_id: str, category: str) -> None:
        gap = PLANTED_MERCHANT_GAPS.get(merchant_id)
        signature_possible = category == "incorrect_amount" or facts["delivery_confirmed"] == "yes"
        if gap and (gap != "signed_by_cardholder" or signature_possible) and self.r.random() < PLANTED_RATE:
            facts[gap] = "no"
        if self.r.random() < PLANTED_RATE:
            facts[PLANTED_SYSTEMIC_GAP] = "unknown"

    # ---------- rendering ----------
    def date(self) -> str:
        return f"{self.r.randint(1, 28):02d}-{self.r.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])}"

    def match_name(self, first: str, last: str) -> str:
        return self.r.choice([f"{first} {last}", f"{first[0]}. {last}", f"{first} {last[0]}.", f"{first.upper()} {last.upper()}"])

    def other_name(self, first: str, last: str) -> str:
        r = self.r
        if r.random() < 0.35:  # near-miss surname: requires actually comparing names
            return f"{first} {SIMILAR_LAST[last]}"
        return f"{r.choice([f for f in FIRST if f != first])} {r.choice([s for s in LAST if s != last])}"

    def carrier(self, facts, first, last, city) -> str | None:
        r, d = self.r, self.date()
        if facts["delivery_confirmed"] == "unknown":
            return None
        lines = [f"Tracking ID: {r.choice(['DL', 'BD', 'XB'])}{r.randint(10**8, 10**9)}"]
        if self.style == "shifted":
            return self._carrier_shifted(facts, first, last, city, lines)
        if facts["delivery_confirmed"] == "yes":
            lines.append(r.choice([
                f"Status: DELIVERED on {d} {r.randint(9, 20)}:{r.randint(10, 59)}.",
                f"Shipment delivered to recipient at the {city} shipping address on {d}.",
                f"Proof of delivery: parcel handed over at the doorstep on {d}.",
                f"Out for delivery -> Delivered ({d}).",
            ]))
        else:
            lines.append(r.choice([
                f"Status: DELIVERY ATTEMPTED - recipient unavailable. Returned to origin hub on {d}.",
                "Shipment could not be delivered: address not found. RTO initiated.",
                f"Last scan: in transit at {city} sorting centre ({d}). No delivery scan recorded.",
                "Courier app marked the parcel delivered, but the GPS scan was 14 km from the shipping address; "
                "the courier's own investigation concluded it was not delivered.",
                "Delivery failed after 3 attempts; parcel returned to seller.",
            ]))
        s = facts["signed_by_cardholder"]
        if s == "yes":
            lines.append(r.choice([f"Signed by: {self.match_name(first, last)}",
                                   f"Recipient signature captured: {self.match_name(first, last)}"]))
        elif s == "no":
            lines.append(r.choice([f"Signed by: {self.other_name(first, last)}",
                                   f"Received by building security ({self.other_name(first, last)})",
                                   "Contactless delivery - no signature collected.",
                                   f"Recipient signature: {self.other_name(first, last)} (neighbour)"]))
        return "\n".join(lines)

    def _carrier_shifted(self, facts, first, last, city, lines) -> str:
        r, d = self.r, self.date()
        if facts["delivery_confirmed"] == "yes":
            lines.append(r.choice([
                f"POD: consignment received at the destination address, {d}.",
                "Parcel left in the safe place (porch) - photo on file.",
                "Consignment status: handed to the addressee.",
            ]))
        else:
            lines.append(r.choice([
                "Consignee not available; shipment held at the facility pending collection.",
                f"Parcel misrouted to {r.choice([c for c in CITIES if c != city])}; final status: lost in transit.",
                "Marked 'delivered' in error by the rider - later corrected to UNDELIVERED.",
            ]))
        s = facts["signed_by_cardholder"]
        if s == "yes":
            lines.append(r.choice([f"POD signature: {self.match_name(first, last)}",
                                   f"Acknowledged by {self.match_name(first, last)} (self)"]))
        elif s == "no":
            lines.append(r.choice([f"POD signature: {self.other_name(first, last)}",
                                   f"Handed to the watchman, {self.other_name(first, last)}",
                                   f"Receiver: family member ({self.other_name(first, last)})"]))
        return "\n".join(lines)

    def customer_messages(self, facts, category) -> str | None:
        r = self.r
        a = facts["customer_acknowledged_receipt"]
        if a == "unknown":
            return None
        if self.style == "shifted":
            opts = (["Package is here but it's not what I want.",
                     "Collected it from the front desk on Tuesday.",
                     "It showed up eventually, I still want my money back."] if a == "yes" else
                    ["Where is my parcel?? Never showed up.",
                     "The delivery person never came to my house.",
                     "Tracking is wrong, I have nothing."])
            return f"From customer ({self.date()}): {r.choice(opts)}"
        if a == "yes":
            opts = [
                "Hi, the parcel came yesterday but I don't want it anymore, please cancel.",
                "Received the box today. I'm still raising a dispute with my bank.",
                "Yes it arrived, but it took 3 weeks which is unacceptable.",
                "Got it on Monday. Not happy with the service.",
            ]
            if category == "not_as_described":
                opts += ["The jacket I received is a completely different colour from the photos.",
                         "Item arrived but it is the older model, not what was advertised."]
            if category == "fraud":
                opts += ["The package did arrive at my place but I never authorised this purchase."]
        else:
            opts = [
                "I have not received anything. Tracking says delivered but nothing is here.",
                "Still waiting for my order, it's been 20 days.",
                "Nothing arrived. I checked with my neighbours and the security desk too.",
                "I did receive an email saying it shipped, but no package ever came.",
                "The app says delivered but I never got it.",
            ]
        msg = r.choice(opts)
        return f"From customer ({self.date()}): {msg}"

    def order_record(self, facts, city) -> str | None:
        r = self.r
        if self.style == "shifted":
            return self._order_shifted(facts, city)
        lines = []
        a = facts["avs_cvv_match"]
        if a == "yes":
            lines.append(r.choice(["AVS result: Y (address + ZIP match) | CVV2: M (match)",
                                   "Address verification: full match; security code: matched"]))
        elif a == "no":
            lines.append(r.choice(["AVS result: N (no match) | CVV2: M (match)",
                                   "AVS result: Z (ZIP only) | CVV2: N (no match)",
                                   "Address verification: partial (postcode only); CVV: matched",
                                   "Address verification: full match; security code: not provided"]))
        i = facts["ip_consistent_with_cardholder"]
        if i == "yes":
            lines.append(r.choice([f"Order IP geolocates to {city}, same city as the billing address.",
                                   f"Device location: {city} (billing address: {city})."]))
        elif i == "no":
            far = r.choice(FAR)
            lines.append(r.choice([f"Order IP geolocates to {far}; billing address is in {city}.",
                                   f"IP flagged as a known VPN/proxy exit node; billing address {city}.",
                                   f"Device location: {r.choice([c for c in CITIES if c != city])} (billing address: {city})."]))
        p = facts["prior_undisputed_orders"]
        if p == "yes":
            n = r.randint(2, 14)
            lines.append(r.choice([f"Account history: {n} previous orders since 20{r.randint(19, 24)}, none disputed.",
                                   f"Same device fingerprint used for {n} earlier purchases with no chargebacks."]))
        elif p == "no":
            lines.append(r.choice([f"Account created {r.randint(3, 50)} minutes before this order; no previous purchases.",
                                   f"Customer has {r.randint(3, 6)} earlier orders, 2 of which were also charged back.",
                                   "First-time customer; guest checkout."]))
        if not lines:
            return None
        return f"Order #{r.randint(10000, 99999)}\n" + "\n".join(lines)

    def _order_shifted(self, facts, city) -> str | None:
        r = self.r
        lines = []
        a = facts["avs_cvv_match"]
        if a == "yes":
            lines.append("Auth response: address_check=PASS, cvc_check=PASS")
        elif a == "no":
            lines.append(r.choice(["Auth response: address_check=FAIL, cvc_check=PASS",
                                   "Auth response: address_check=UNAVAILABLE, cvc_check=PASS",
                                   "Auth response: address_check=PASS, cvc_check=FAIL"]))
        i = facts["ip_consistent_with_cardholder"]
        if i == "yes":
            lines.append(f"Checkout session from an ISP in {city}; cardholder resides in {city}.")
        elif i == "no":
            lines.append(r.choice(["Checkout session originated from a Tor exit node.",
                                   f"Checkout session from an ISP in {r.choice(FAR)}; cardholder resides in {city}."]))
        p = facts["prior_undisputed_orders"]
        if p == "yes":
            lines.append(f"Customer since 20{r.randint(19, 23)} with {r.randint(2, 14)} fulfilled orders and zero disputes.")
        elif p == "no":
            lines.append(r.choice(["Brand-new account, email address created the same day.",
                                   f"Customer has a history of {r.randint(2, 5)} prior disputes."]))
        if not lines:
            return None
        return f"Order #{r.randint(10000, 99999)}\n" + "\n".join(lines)

    def fulfilment(self, facts) -> str | None:
        r = self.r
        m = facts["item_matches_description"]
        if m == "unknown":
            return None
        sku = f"SKU-{r.randint(1000, 9999)}"
        if self.style == "shifted":
            if m == "yes":
                return "Outbound scan confirms the listed model and colour; packing photo archived."
            return r.choice(["Listed: 128GB variant. Shipped: 64GB variant.",
                             "Colour mismatch acknowledged by warehouse (listed navy, shipped black)."])
        if m == "yes":
            return r.choice([f"Picked {sku}, matches listing {sku}; QC photo taken before dispatch.",
                             "Warehouse QC: item, colour and size verified against the listing."])
        other = f"SKU-{r.randint(1000, 9999)}"
        return r.choice([f"Picked {other} (listing is {sku}).",
                         "Stock-out on the listed variant; substituted with a similar model.",
                         "QC note: shipped item was the 2022 model, not the 2024 model shown in the listing."])

    def refund(self, facts) -> str | None:
        r, d = self.r, self.date()
        v = facts["return_or_refund_offered"]
        if v == "unknown":
            return None
        if self.style == "shifted":
            if v == "yes":
                return r.choice(["We sent a prepaid return label; unused as of today.",
                                 "Customer was told a replacement or refund was available; they chose to dispute instead."])
            return r.choice(["Return window lapsed; refund refused.",
                             "The customer's refund email was never replied to."])
        if v == "yes":
            return r.choice([f"Return label emailed to the customer on {d}; the customer did not use it.",
                             "Offered a full refund on return (policy accepted at checkout); no response from customer."])
        return r.choice([f"Customer requested a return on {d}; request declined as outside the policy window.",
                         "No return or refund was offered.",
                         "Refund request is still unanswered by the merchant."])

    def invoice(self, facts, amount, first, last, category) -> str | None:
        r = self.r
        lines = []
        a = facts["amount_matches_agreement"]
        if self.style == "shifted" and a != "unknown":
            agreed = amount if a == "yes" else int(amount * r.choice([0.6, 0.75, 0.9]))
            lines.append(f"Quoted INR {agreed:,}; captured INR {amount:,}.")
        elif a == "yes":
            lines.append(r.choice([f"Invoice total Rs {amount:,} (incl. shipping); card charged Rs {amount:,}.",
                                   f"Customer approved a quote of Rs {amount:,}; amount charged Rs {amount:,}."]))
        elif a == "no":
            agreed = int(amount * r.choice([0.6, 0.75, 0.9]))
            lines.append(r.choice([f"Invoice total Rs {agreed:,}; card charged Rs {amount:,} (duplicate shipping fee).",
                                   f"Agreed price Rs {agreed:,}; charged Rs {amount:,} after a currency markup."]))
        if category == "incorrect_amount":
            s = facts["signed_by_cardholder"]
            if s == "yes" and self.style == "shifted":
                lines.append(f"Signature on invoice: {self.match_name(first, last)} (account holder)")
            elif s == "no" and self.style == "shifted":
                lines.append(f"Signature on invoice: {self.other_name(first, last)} (store staff)")
            elif s == "yes":
                lines.append(f"Invoice signed by customer: {self.match_name(first, last)}")
            elif s == "no":
                lines.append(r.choice([f"Invoice signed by: {self.other_name(first, last)}", "Invoice was not signed."]))
        return "\n".join(lines) if lines else None

    # ---------- case ----------
    def case(self, idx: int, split: str) -> dict:
        r = self.r
        code = r.choices(list(RC_WEIGHTS), weights=list(RC_WEIGHTS.values()))[0]
        rc = self.reason_codes[code]
        category = rc["category"]
        merchant = r.choice(self.merchants)
        first, last, city = r.choice(FIRST), r.choice(LAST), r.choice(CITIES)
        facts = self.sample_facts(rc["relevant_facts"], category)
        if self.planted:
            self.plant(facts, merchant["merchant_id"], category)
        amount = int(min(150_000, max(150, round(math.exp(r.gauss(math.log(3000), 1.1)), -1))))

        docs = []
        for doc_type, text in [
            ("carrier_record", self.carrier(facts, first, last, city) if category != "incorrect_amount" else None),
            ("customer_messages", self.customer_messages(facts, category)),
            ("order_record", self.order_record(facts, city)),
            ("fulfilment_record", self.fulfilment(facts)),
            ("refund_record", self.refund(facts)),
            ("invoice", self.invoice(facts, amount, first, last, category)),
        ]:
            if text:
                docs.append({"doc_type": doc_type, "text": text})
        r.shuffle(docs)

        # Facts the documents don't express are unknown to everyone, including the issuer.
        if category == "incorrect_amount" and facts["delivery_confirmed"] != "unknown":
            facts["delivery_confirmed"] = "unknown"
        p_win = self.true_p_win(category, facts, merchant)
        return {
            "case_id": f"{split}_{idx:05d}",
            "merchant_id": merchant["merchant_id"],
            "reason_code": code,
            "category": category,
            "amount_inr": amount,
            "cardholder_name": f"{first} {last}",
            "response_deadline_days_left": r.randint(0, 25),
            "is_repeat_dispute": r.random() < 0.1,
            "documents": docs,
            "true_facts": facts,
            "true_p_win": round(p_win, 4),
            "outcome": "won" if r.random() < p_win else "lost",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--n-portfolio", type=int, default=1200)
    args = ap.parse_args()

    splits = [("train", args.n_train, "dev"), ("val", args.n_val, "dev"),
              ("test", args.n_test, "dev"), ("test_shifted", args.n_test, "shifted"),
              ("portfolio", args.n_portfolio, "dev")]
    for offset, (split, n, style) in enumerate(splits):
        gen = Gen(args.seed * 100 + offset, style=style, planted=split == "portfolio")
        rows = [gen.case(i, split) for i in range(n)]
        path = DATA / f"disputes_{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
        won = sum(row["outcome"] == "won" for row in rows)
        print(f"{split}: {n} cases, win rate {won / n:.1%} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
