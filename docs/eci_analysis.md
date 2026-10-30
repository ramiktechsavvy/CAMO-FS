# Explanation Consistency Index (ECI)

ECI was calculated from the five saved SHAP fold rankings; SHAP was not recomputed.

For each of the 10 fold pairs, Kendall tau-b is computed over the selected features' ordinal SHAP-importance ranks. ECI is `(mean pairwise Kendall tau + 1) / 2`, which maps concordance to [0, 1]. Ties are handled with tau-b. For a subset with fewer than two features, ECI is defined as 1 because no ordering disagreement is estimable.

Overall ECI: **0.996387**

| Operational representation | ECI |
|---|---:|
| EARLY | 0.995295 |
| TRANSITION | 0.996632 |
| LATE | 0.996044 |

These labels refer to July 5, date-to-date change, and July 14 representations. They are not validated biological growth stages.

Pairwise coefficients are saved in `results/eci/pairwise_kendall_tau.csv`; summary values are in `results/eci/eci_summary.json`.
