# Portfolio Intelligence — Root Cause Report

Total records: 57  |  Resolved: 56  |  Excluded (unknown outcome): 1
Won: 33  |  Lost: 23  |  Overall loss rate: 0.411

*Note: `total` in the tables below counts every record for that reason code / merchant, including unknown-outcome ones — it is not the same as `won + lost`. `total == won + lost + unknown` always holds.*

## Highlights

- 📊 **Top systemic weakness:** customer_communication_log — present in 7 of 23 lost disputes (30%).
- 🚨 **Highest loss rate:** mch_03 (80%)
- 💰 **Highest financial exposure:** mch_05 (₹100,000 lost)

## Missing-evidence frequency among losses

| evidence gap | count | % of losses |
|---|---|---|
| customer_communication_log | 7 | 30% |
| tracking_number | 6 | 26% |
| ip_geolocation | 5 | 22% |
| auth_data | 4 | 17% |
| delivery_proof | 4 | 17% |
| return_policy_acceptance | 4 | 17% |
| (no missing evidence recorded — non-evidence loss) | 3 | 13% |
| product_description_match | 2 | 9% |
| terms_acceptance | 1 | 4% |

## By reason code

| reason_code | total | won | lost | unknown | loss_rate |
|---|---|---|---|---|---|
| 13.1 | 15 | 7 | 8 | 0 | 0.533 |
| 4837 | 5 | 2 | 3 | 0 | 0.6 |
| 13.3 | 7 | 4 | 3 | 0 | 0.429 |
| F29 | 5 | 2 | 3 | 0 | 0.6 |
| 10.4 | 6 | 2 | 3 | 1 | 0.6 |
| 4853 | 5 | 3 | 2 | 0 | 0.4 |
| 12.5 | 8 | 7 | 1 | 0 | 0.125 |
| C08 | 6 | 6 | 0 | 0 | 0.0 |

## By merchant

| merchant_id | total | won | lost | unknown | loss_rate | amount_lost_inr |
|---|---|---|---|---|---|---|
| mch_05 | 9 | 4 | 5 | 0 | 0.556 | ₹100,000 |
| mch_07 | 6 | 4 | 2 | 0 | 0.333 | ₹95,900 |
| mch_04 | 13 | 8 | 5 | 0 | 0.385 | ₹31,600 |
| mch_03 | 5 | 1 | 4 | 0 | 0.8 | ₹18,700 |
| mch_06 | 5 | 3 | 2 | 0 | 0.4 | ₹11,600 |
| mch_01 | 5 | 3 | 2 | 0 | 0.4 | ₹7,000 |
| mch_02 | 9 | 7 | 2 | 0 | 0.222 | ₹5,700 |
| mch_08 | 5 | 3 | 1 | 1 | 0.25 | ₹3,800 |