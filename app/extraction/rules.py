"""
Deterministic rule-based fact extractor.

Two jobs: the degraded path when the LLM is unavailable, and the baseline
the LLM extractor has to beat in the eval. It is written to be competent
(negation handling, name comparison, amount comparison), not a strawman,
because an ablation against a weak baseline proves nothing.
"""
from __future__ import annotations

import re

from app.domain import DocType, EvidenceDocument, EvidenceFacts, Tri

_NEG_DELIVERY = re.compile(
    r"could not be delivered|not delivered|delivery (?:failed|attempted)|no delivery scan|returned to|rto\b|in transit",
    re.I,
)
_POS_DELIVERY = re.compile(r"\bdelivered\b|handed over", re.I)
_SIGNED = re.compile(r"(?:signed by|signature captured|signature|signed by customer)\s*:\s*([^\n(]+)", re.I)
_NO_SIGNATURE = re.compile(r"no signature|not signed|received by building security", re.I)
_NEG_RECEIPT = re.compile(r"not received|never (?:got|came|arrived)|no package|nothing (?:arrived|is here)|still waiting", re.I)
_POS_RECEIPT = re.compile(r"\b(?:came|arrived|received|got it)\b", re.I)
_AMOUNTS = re.compile(r"Rs\s*([\d,]+)")


def _docs(documents: list[EvidenceDocument], doc_type: DocType) -> str:
    return "\n".join(d.text for d in documents if d.doc_type == doc_type)


def _names_match(signed: str, cardholder: str) -> bool:
    """First initial + surname must match, tolerating 'P. Nair' / 'Priya N.' forms."""
    s = [t.strip(".").lower() for t in signed.split() if t.strip(".")]
    c = [t.lower() for t in cardholder.split()]
    if len(s) < 2 or len(c) < 2:
        return False
    first_ok = s[0][0] == c[0][0] and (len(s[0]) == 1 or s[0] == c[0])
    last_ok = s[-1] == c[-1] or (len(s[-1]) == 1 and s[-1] == c[-1][0])
    return first_ok and last_ok


def _delivery(carrier: str) -> Tri:
    if not carrier:
        return Tri.UNKNOWN
    if _NEG_DELIVERY.search(carrier):
        return Tri.NO
    if _POS_DELIVERY.search(carrier):
        return Tri.YES
    return Tri.UNKNOWN


def _signature(text: str, cardholder: str) -> Tri:
    if _NO_SIGNATURE.search(text):
        return Tri.NO
    m = _SIGNED.search(text)
    if not m:
        return Tri.UNKNOWN
    return Tri.YES if _names_match(m.group(1).strip(), cardholder) else Tri.NO


def _receipt(messages: str) -> Tri:
    if not messages:
        return Tri.UNKNOWN
    if _NEG_RECEIPT.search(messages):
        return Tri.NO
    if _POS_RECEIPT.search(messages):
        return Tri.YES
    return Tri.UNKNOWN


def _avs_cvv(order: str) -> Tri:
    line = next((ln for ln in order.splitlines() if re.search(r"AVS|address verification", ln, re.I)), "")
    if not line:
        return Tri.UNKNOWN
    avs_ok = bool(re.search(r"AVS result: Y|full match", line, re.I))
    cvv_ok = bool(re.search(r"CVV2?: M|(?:CVV|security code): matched", line, re.I))
    return Tri.YES if avs_ok and cvv_ok else Tri.NO


def _ip(order: str) -> Tri:
    if re.search(r"VPN|proxy", order, re.I):
        return Tri.NO
    m = re.search(r"(?:geolocates to|Device location:)\s*([^;(,.]+).*?billing address(?: is in|:)?\s*([A-Za-z ]+)", order, re.I)
    if not m:
        if re.search(r"same city as the billing", order, re.I):
            return Tri.YES
        return Tri.UNKNOWN
    if re.search(r"same city as the billing", order, re.I):
        return Tri.YES
    return Tri.YES if m.group(1).strip().lower() == m.group(2).strip().lower() else Tri.NO


def _prior(order: str) -> Tri:
    if re.search(r"none disputed|no chargebacks", order, re.I):
        return Tri.YES
    if re.search(r"no previous purchases|first-time customer|also charged back", order, re.I):
        return Tri.NO
    return Tri.UNKNOWN


def _item_match(fulfilment: str) -> Tri:
    if not fulfilment:
        return Tri.UNKNOWN
    skus = re.findall(r"SKU-\d+", fulfilment)
    if len(skus) >= 2:
        return Tri.YES if len(set(skus)) == 1 else Tri.NO
    if re.search(r"substitut|not the .* shown|different", fulfilment, re.I):
        return Tri.NO
    if re.search(r"verified against|matches listing", fulfilment, re.I):
        return Tri.YES
    return Tri.UNKNOWN


def _refund(refund: str) -> Tri:
    if not refund:
        return Tri.UNKNOWN
    if re.search(r"declined|no return or refund|unanswered", refund, re.I):
        return Tri.NO
    if re.search(r"offered|return label", refund, re.I):
        return Tri.YES
    return Tri.UNKNOWN


def _amount(invoice: str) -> Tri:
    amounts = [int(a.replace(",", "")) for a in _AMOUNTS.findall(invoice)]
    if len(amounts) < 2:
        return Tri.UNKNOWN
    return Tri.YES if amounts[0] == amounts[1] else Tri.NO


def extract_with_rules(documents: list[EvidenceDocument], cardholder_name: str) -> EvidenceFacts:
    carrier = _docs(documents, DocType.CARRIER_RECORD)
    invoice = _docs(documents, DocType.INVOICE)
    order = _docs(documents, DocType.ORDER_RECORD)

    signature = _signature(carrier, cardholder_name) if carrier else Tri.UNKNOWN
    if signature == Tri.UNKNOWN and invoice:
        signature = _signature(invoice, cardholder_name)

    return EvidenceFacts(
        delivery_confirmed=_delivery(carrier),
        signed_by_cardholder=signature,
        customer_acknowledged_receipt=_receipt(_docs(documents, DocType.CUSTOMER_MESSAGES)),
        item_matches_description=_item_match(_docs(documents, DocType.FULFILMENT_RECORD)),
        return_or_refund_offered=_refund(_docs(documents, DocType.REFUND_RECORD)),
        avs_cvv_match=_avs_cvv(order),
        ip_consistent_with_cardholder=_ip(order),
        prior_undisputed_orders=_prior(order),
        amount_matches_agreement=_amount(invoice),
    )


# Every line of a document the rules understand matches one of these. A
# document with any line left over contains content the rules could not
# interpret, so its unknown facts may be parse failures rather than genuine
# absences. A fully understood document that never mentions a fact is a
# genuine unknown.
_BOILERPLATE = re.compile(r"^\s*(?:tracking id:|order #|from customer \([^)]*\):\s*$)", re.I)
_LINE_PATTERNS: dict[DocType, list[re.Pattern]] = {
    DocType.CARRIER_RECORD: [_NEG_DELIVERY, _POS_DELIVERY, _SIGNED, _NO_SIGNATURE],
    DocType.CUSTOMER_MESSAGES: [_NEG_RECEIPT, _POS_RECEIPT],
    DocType.ORDER_RECORD: [re.compile(r"AVS|address verification|geolocates|device location|VPN|proxy|none disputed|"
                                      r"no chargebacks|no previous purchases|first-time customer|charged back", re.I)],
    DocType.FULFILMENT_RECORD: [re.compile(r"SKU-\d+|substitut|verified against|not the .* shown", re.I)],
    DocType.REFUND_RECORD: [re.compile(r"declined|no return or refund|unanswered|offered|return label", re.I)],
    DocType.INVOICE: [_AMOUNTS, _SIGNED, re.compile(r"not signed", re.I)],
}


def unreadable_doc_types(documents: list[EvidenceDocument]) -> set[DocType]:
    """Document types containing at least one line no rule could interpret."""
    unreadable = set()
    for doc in documents:
        for line in doc.text.splitlines():
            if not line.strip() or _BOILERPLATE.match(line):
                continue
            if not any(p.search(line) for p in _LINE_PATTERNS[doc.doc_type]):
                unreadable.add(doc.doc_type)
                break
    return unreadable
