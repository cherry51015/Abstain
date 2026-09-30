# Portfolio diagnosis: planted-problem check

1200 disputes from the `portfolio` split were evaluated and resolved through the HTTP API (`scripts/portfolio_demo.py`). The split has planted process problems; the reports must find them and nothing else.

| check | result | detected |
|---|---|---|
| systemic gap recovered | PASS | `['prior_undisputed_orders']` |
| mch_04: planted `signed_by_cardholder` flagged as merchant-specific | PASS | `['signed_by_cardholder']` |
| mch_06: planted `return_or_refund_offered` flagged as merchant-specific | PASS | `['return_or_refund_offered']` |
| no merchant-specific false alarms elsewhere | PASS | `none` |

---

# Portfolio

Resolved disputes: **1200** · lost: **700**

## Systemic gaps

- **`prior_undisputed_orders`** is missing in 80% of relevant disputes, in most disputes of 8 separate merchants (mch_01, mch_02, mch_03, mch_04, mch_05, mch_06, mch_07, mch_08), and disputes without it lose 69% of the time vs 40% with it: fix the shared evidence pipeline, not individual merchants.

## Merchant primary weaknesses (benchmarked against the rest of the portfolio)

| merchant | losses | primary weakness | weak in (this merchant) | rest of portfolio | in losses | verdict |
|---|---|---|---|---|---|---|
| mch_01 | 77 | `prior_undisputed_orders` | 92% of 48 | 92% | 27/30 | systemic |
| mch_02 | 93 | `prior_undisputed_orders` | 89% of 54 | 92% | 40/42 | systemic |
| mch_03 | 68 | `prior_undisputed_orders` | 91% of 57 | 92% | 32/33 | systemic |
| mch_04 | 104 | `signed_by_cardholder` | 88% of 61 | 49% | 32/35 | merchant-specific |
| mch_05 | 84 | `prior_undisputed_orders` | 96% of 57 | 91% | 37/38 | systemic |
| mch_06 | 90 | `return_or_refund_offered` | 95% of 21 | 54% | 16/17 | merchant-specific |
| mch_07 | 102 | `prior_undisputed_orders` | 92% of 77 | 92% | 47/50 | systemic |
| mch_08 | 82 | `prior_undisputed_orders` | 88% of 58 | 92% | 44/48 | systemic |

## Why disputes are lost

*Missing* = no document covered the fact (collect it). *Adverse* = the document says no (fix operations).

| fact | missing in losses | adverse in losses |
|---|---|---|
| `signed_by_cardholder` | 375 | 114 |
| `delivery_confirmed` | 92 | 239 |
| `prior_undisputed_orders` | 264 | 0 |
| `ip_consistent_with_cardholder` | 67 | 162 |
| `avs_cvv_match` | 61 | 152 |
| `customer_acknowledged_receipt` | 0 | 89 |

## Merchants

| merchant | resolved | loss rate | ₹ lost | network-ratio proximity |
|---|---|---|---|---|
| mch_01 | 157 | 0.49 | 478,670 | 39% |
| mch_02 | 139 | 0.669 | 525,340 | 91% |
| mch_03 | 143 | 0.476 | 434,890 | 24% |
| mch_04 | 151 | 0.689 | 676,960 | 98% |
| mch_05 | 151 | 0.556 | 497,190 | 61% |
| mch_06 | 142 | 0.634 | 364,530 | 106% |
| mch_07 | 175 | 0.583 | 493,400 | 11% |
| mch_08 | 142 | 0.577 | 496,720 | 78% |

## Decisions that were contested

- Contested: 588, won: 347 (win rate 0.59)
- Net recovered after contest costs: **₹2,036,480**

## P(win) calibration on resolved cases

- n = 1200, Brier = 0.1395, ECE = 0.024

---

# Merchant mch_04

Resolved disputes: **151** · lost: **104**

## Primary weakness

- **`signed_by_cardholder`** was weak (missing or adverse) in **32 of 35** losses where it mattered.
- Weak in 88% of this merchant's disputes vs 49% for the rest of the portfolio (z = 5.84): **merchant-specific**.

| fact | this merchant | rest of portfolio | z | verdict |
|---|---|---|---|---|
| `signed_by_cardholder` | 88% of 61 | 49% | 5.84 | merchant-specific |

## Why disputes are lost

*Missing* = no document covered the fact (collect it). *Adverse* = the document says no (fix operations).

| fact | missing in losses | adverse in losses |
|---|---|---|
| `signed_by_cardholder` | 50 | 30 |
| `delivery_confirmed` | 14 | 34 |
| `prior_undisputed_orders` | 39 | 0 |
| `ip_consistent_with_cardholder` | 12 | 25 |
| `avs_cvv_match` | 9 | 24 |
| `customer_acknowledged_receipt` | 0 | 13 |

## Decisions that were contested

- Contested: 60, won: 31 (win rate 0.517)
- Net recovered after contest costs: **₹292,510**

## P(win) calibration on resolved cases

- n = 151, Brier = 0.1336, ECE = 0.0712

---

# Merchant mch_06

Resolved disputes: **142** · lost: **90**

## Primary weakness

- **`return_or_refund_offered`** was weak (missing or adverse) in **16 of 17** losses where it mattered.
- Weak in 95% of this merchant's disputes vs 54% for the rest of the portfolio (z = 3.63): **merchant-specific**.

| fact | this merchant | rest of portfolio | z | verdict |
|---|---|---|---|---|
| `return_or_refund_offered` | 95% of 21 | 54% | 3.63 | merchant-specific |

## Why disputes are lost

*Missing* = no document covered the fact (collect it). *Adverse* = the document says no (fix operations).

| fact | missing in losses | adverse in losses |
|---|---|---|
| `signed_by_cardholder` | 36 | 17 |
| `prior_undisputed_orders` | 36 | 0 |
| `delivery_confirmed` | 10 | 18 |
| `avs_cvv_match` | 8 | 18 |
| `ip_consistent_with_cardholder` | 9 | 17 |
| `return_or_refund_offered` | 0 | 16 |

## Decisions that were contested

- Contested: 84, won: 41 (win rate 0.488)
- Net recovered after contest costs: **₹242,910**

## P(win) calibration on resolved cases

- n = 142, Brier = 0.1514, ECE = 0.077
