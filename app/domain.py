"""
Core domain types shared by every layer.

The contract between the AI side and the decision side is `EvidenceFacts`:
a fixed set of tri-state facts (yes / no / unknown) that an extractor reads
out of the raw evidence documents. Facts are verifiable against labels, so
extraction quality can be measured directly instead of being hidden inside
an opaque "evidence strength" score.
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Action(str, Enum):
    CONTEST = "CONTEST"
    CONCEDE = "CONCEDE"
    ESCALATE = "ESCALATE"


class Tri(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class Category(str, Enum):
    NOT_RECEIVED = "not_received"
    FRAUD = "fraud"
    NOT_AS_DESCRIBED = "not_as_described"
    INCORRECT_AMOUNT = "incorrect_amount"


# Fact name -> definition. The definitions are sent verbatim to the LLM, so
# they are written as precise yes/no questions from the merchant's side.
FACT_DEFINITIONS: dict[str, str] = {
    "delivery_confirmed": "Does a carrier record confirm the item was delivered to the shipping address?",
    "signed_by_cardholder": "Was the delivery or invoice signed by the cardholder themself? 'no' if someone else signed or no signature was collected.",
    "customer_acknowledged_receipt": "Do the customer's own messages acknowledge that the item arrived?",
    "item_matches_description": "Does the fulfilment record show the shipped item matched the product listing?",
    "return_or_refund_offered": "Did the merchant offer a return or refund that the customer did not take up?",
    "avs_cvv_match": "Does the authorization data show BOTH the address check (AVS) and CVV matched?",
    "ip_consistent_with_cardholder": "Is the order's IP/device location consistent with the cardholder's billing location?",
    "prior_undisputed_orders": "Does the same account/device have earlier orders that were never disputed?",
    "amount_matches_agreement": "Does the charged amount match the amount the customer agreed to (invoice/terms)?",
}
FACT_NAMES: tuple[str, ...] = tuple(FACT_DEFINITIONS)


class DocType(str, Enum):
    CARRIER_RECORD = "carrier_record"
    CUSTOMER_MESSAGES = "customer_messages"
    ORDER_RECORD = "order_record"
    FULFILMENT_RECORD = "fulfilment_record"
    REFUND_RECORD = "refund_record"
    INVOICE = "invoice"


# Which document types can answer each fact. Used to tell "no document
# covers this" (a true unknown) apart from "a document exists but the rules
# could not read it" (a gap worth an LLM call).
FACT_SOURCES: dict[str, tuple[DocType, ...]] = {
    "delivery_confirmed": (DocType.CARRIER_RECORD,),
    "signed_by_cardholder": (DocType.CARRIER_RECORD, DocType.INVOICE),
    "customer_acknowledged_receipt": (DocType.CUSTOMER_MESSAGES,),
    "item_matches_description": (DocType.FULFILMENT_RECORD,),
    "return_or_refund_offered": (DocType.REFUND_RECORD,),
    "avs_cvv_match": (DocType.ORDER_RECORD,),
    "ip_consistent_with_cardholder": (DocType.ORDER_RECORD,),
    "prior_undisputed_orders": (DocType.ORDER_RECORD,),
    "amount_matches_agreement": (DocType.INVOICE,),
}


# A fact that can only exist once another fact holds: a delivery signature
# needs a delivery. Where the prerequisite is relevant and not met, the fact
# is structurally absent, not a collection gap, and analytics must not count
# it (doing so confounds "no signature" with "no delivery").
FACT_PREREQUISITES: dict[str, tuple[str, Tri]] = {
    "signed_by_cardholder": ("delivery_confirmed", Tri.YES),
}


def fact_applicable(fact: str, facts: dict[str, str], relevant: list[str]) -> bool:
    """False when the fact's prerequisite is relevant here and not met."""
    pre = FACT_PREREQUISITES.get(fact)
    if pre is None or pre[0] not in relevant:
        return True
    return facts.get(pre[0], Tri.UNKNOWN.value) == pre[1].value


class EvidenceFacts(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)

    delivery_confirmed: Tri = Tri.UNKNOWN
    signed_by_cardholder: Tri = Tri.UNKNOWN
    customer_acknowledged_receipt: Tri = Tri.UNKNOWN
    item_matches_description: Tri = Tri.UNKNOWN
    return_or_refund_offered: Tri = Tri.UNKNOWN
    avs_cvv_match: Tri = Tri.UNKNOWN
    ip_consistent_with_cardholder: Tri = Tri.UNKNOWN
    prior_undisputed_orders: Tri = Tri.UNKNOWN
    amount_matches_agreement: Tri = Tri.UNKNOWN

    def get(self, name: str) -> Tri:
        return getattr(self, name)


class EvidenceDocument(BaseModel):
    doc_type: DocType
    text: str = Field(min_length=1, max_length=8000)


class ReasonCode(BaseModel):
    code: str
    network: str
    label: str
    category: Category
    relevant_facts: list[str]


class Merchant(BaseModel):
    merchant_id: str
    name: str
    historical_win_rate: float = Field(ge=0.0, le=1.0)
    current_chargeback_rate_pct: float = Field(ge=0.0)
    network_threshold_pct: float = Field(gt=0.0)
    ops_cost_per_contest_inr: Decimal = Field(gt=0)
    risk_tolerance: Literal["aggressive", "moderate", "conservative"] = "moderate"

    @property
    def threshold_proximity(self) -> float:
        return self.current_chargeback_rate_pct / self.network_threshold_pct


class DisputeCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=64)
    merchant_id: str
    reason_code: str
    amount_inr: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    cardholder_name: str = Field(min_length=1, max_length=120)
    response_deadline_days_left: int = Field(ge=0, le=120)
    is_repeat_dispute: bool = False
    documents: list[EvidenceDocument] = Field(default_factory=list, max_length=20)
