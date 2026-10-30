# CAMO-FS Step 0 Dataset and Schema Audit

**Audit scope:** direct read of the local ZIP member plus comparison with the official UCI Dataset #525 metadata. The archive was read in place; it was not extracted, and no derived dataset was generated. The only project files created for this task are this report and [`data_audit.json`](data_audit.json).

**Source archive:** `data/raw/crop+mapping+using+fused+optical+radar+data+set.zip`  
**Archive member:** `WinnipegDataset.txt`  
**SHA-256 (ZIP bytes):** `6068727400df7cc12b397d572c9293220cba2515aed6d7573f40cf009a571c3a`

The authoritative metadata is the [UCI Machine Learning Repository page for Dataset #525](https://archive.ics.uci.edu/dataset/525/crop%2Bmapping%2Busing%2Bfused%2Boptical%2Bradar%2Bdata%2Bset). It identifies 325,834 instances; 175 attributes including the class; two dates (July 5 and July 14, 2012); 49 radar features per date; and 38 optical features per date. Its variable information explicitly maps all `f1`–`f174` to sensor, measurement name, and date.

## A. Directly verified from the dataset

### Dimensions, columns, target, and types

- **Rows:** 325,834 data records.
- **Columns:** 175 total.
- **Target:** `label`.
- **Predictors:** 174, named `f1` through `f174` in that order.
- **Exact column order:** `label`, then `f1`, `f2`, …, `f174`. The exhaustive ordered names are in the JSON report.
- **Data types:** all columns are numeric. There are 169 `float64` and 6 `int64` columns. The `int64` columns are `label`, `f137`, `f138`, `f139`, `f140`, and `f141`; the other 169 predictors are `float64`.
- **Missing values / NaN:** zero missing values and zero parsed NaNs.
- **Positive/negative infinity:** zero of either sign.
- **Malformed rows:** none; all records have the same 175-field structure.
- **Duplicate records:** zero byte-identical data lines by SHA-256 digest (header excluded); zero duplicates in a second parsed-row hash check.

### Target labels and distribution

The local file contains label values 1–7, with the following counts. Class names follow UCI's metadata.

| Label | UCI class | Count | Share |
|---:|---|---:|---:|
| 1 | Corn | 39,162 | 12.0190% |
| 2 | Peas | 3,598 | 1.1042% |
| 3 | Canola | 75,673 | 23.2244% |
| 4 | Soybeans | 74,067 | 22.7315% |
| 5 | Oats | 47,117 | 14.4604% |
| 6 | Wheat | 85,074 | 26.1096% |
| 7 | Broadleaf | 1,143 | 0.3508% |
| **Total** |  | **325,834** | **100%** |

### Value ranges and signs

Across predictors, the observed finite range is **−45,280 (`f157`) to 1,105,900 (`f154`)**. Negative values occur in 69 of 174 predictors; exact zero occurs in 44 predictors. Therefore the raw predictor matrix is not uniformly positive.

| UCI block | Columns | Count | Min | Max | Negative values | Zero values |
|---|---|---:|---:|---:|---:|---:|
| Radar, July 5 | `f1`–`f49` | 49 | −34.308 | 73.211 | 5,891,982 | 1,370 |
| Radar, July 14 | `f50`–`f98` | 49 | −35.604 | 62.363 | 5,816,588 | 5,552 |
| Optical, July 5 | `f99`–`f136` | 38 | −5.889 | 556 | 565,703 | 541,557 |
| Optical, July 14 | `f137`–`f174` | 38 | −45,280 | 1,105,900 | 464,419 | 870,673 |

The complete per-feature minimum, maximum, negative count, zero count, and positive count are in `data_audit.json`.

## B. Verified from official UCI metadata

UCI describes a **bi-temporal** fused optical-radar dataset acquired on **July 5 and July 14, 2012**. It reports 175 attributes **including** the class attribute. In the local data file this is exactly `label` plus 174 predictors; 175 is not the number of predictor features.

| Modality and date | Predictor columns | Count |
|---|---|---:|
| Radar, July 5 | `f1`–`f49` | 49 |
| Radar, July 14 | `f50`–`f98` | 49 |
| Optical, July 5 | `f99`–`f136` | 38 |
| Optical, July 14 | `f137`–`f174` | 38 |

UCI's variable metadata names each measured variable with a date suffix (for example, `sigHH_Rad05July`, `sigHH_Rad14July`, `B_Opt05July`, and `B_Opt14July`). The ordering is **sensor/date block order**, not chronological order across all columns: July 5 radar, July 14 radar, July 5 optical, July 14 optical.

UCI provides no three-stage phenology labels, no stage date thresholds, and no mapping to the paper's 58/62/55 partition.

## C. Derived calculations

### Cross-date pairing and proposed `f1↔f88` offset

Pairing by the same UCI measurement name and modality gives **87 valid same-variable pairs**:

- Radar: `f1↔f50`, `f2↔f51`, …, `f49↔f98` (offset **49**).
- Optical: `f99↔f137`, `f100↔f138`, …, `f136↔f174` (offset **38**).

For every one of these 87 pairs, UCI's metadata gives the same base measurement name and sensor with only the acquisition-date suffix changed. The JSON contains the complete per-pair mapping.

The proposed offset pairing `f1↔f88`, `f2↔f89`, …, `f87↔f174` is **incorrect: 0 of 87 proposed pairs match the UCI same-variable cross-date mapping**. For example, `f1` is `sigHH_Rad05July`, while `f88` is `paulgamma_Rad14July`; `f50` is radar `sigHH` on July 14, while `f137` is optical `B` on July 14.

### Constructibility of a two-date derived representation

The raw data do support construction of three **representations** with 87 aligned channels each:

1. `EARLY`: the July 5 measurements (`f1`–`f49` and `f99`–`f136`), arranged in a consistent radar-then-optical channel order.
2. `TRANSITION`: July 14 minus July 5 for each same-variable pair above.
3. `LATE`: the July 14 measurements (`f50`–`f98` and `f137`–`f174`), in the corresponding channel order.

These names can describe the first acquisition, the between-date change, and the second acquisition only. **They are not verified phenological stages.** The dataset has two acquisition dates, and the subtraction-derived `TRANSITION` is a change representation, not a third acquisition or a real growth stage. It must not be presented as evidence for an Early/Transition/Late phenology partition.

### Positivity audit for optional log ratios

A feature-wise `log(July14 / July05)` requires both matched values to be strictly positive for each row. Of the 87 valid measurement pairs:

- **40 pairs** are strictly positive on both dates for all 325,834 records.
- **47 pairs** have at least one non-positive operand in at least one record; these pairs do not support an unmasked, full-data log ratio.
- Across the other 47 pairs, the number of rows with both operands positive ranges from 0 to 325,833.

Thus a global unmasked log-ratio transform is not valid across all 87 predictors. The per-pair valid-row count is recorded in the JSON. Any use of log ratios would need to be restricted to globally positive pairs or define and document a separate treatment for non-positive values; no such transformation was applied in this audit.

## D. Assumptions and audit methods

- Feature identities were taken from UCI's official Dataset #525 variable information, not inferred from correlations.
- Same-variable pairs were formed only where UCI documents the same measurement name and modality at both dates.
- Ranges and sign counts were computed from all parsed finite predictor values in the local ZIP member.
- Duplicate detection found no byte-identical raw data lines and no duplicate parsed rows by pandas row hash. Hash collisions are theoretically possible; the separate raw-line SHA-256 check also returned zero duplicates.
- `EARLY`, `TRANSITION`, and `LATE` above are operational labels for a proposed two-date representation, not assertions about plant phenology.

## E. Unresolved issues

1. The official paper's 58/62/55 stage mapping remains unsupported by the raw UCI schema: neither date thresholds nor exact feature IDs are present in UCI metadata or the local dataset. Its stage totals sum to 175, while the raw input has 174 predictors plus a label.
2. The two UCI dates alone do not establish Vegetative, Flowering/Peak Canopy, or Maturity stages. Independent phenological evidence would be needed to make those claims.
3. The dataset documentation does not establish whether July 5 and July 14 correspond to particular growth stages across the represented crop classes.
4. The report does not validate the paper's sampling, preprocessing, or model-result claims; it is limited to the source archive and schema audit requested here.

## Final decision

**PASS WITH CAVEATS** — the raw data support aligned July 5 and July 14 snapshots and a mathematically defined between-date difference for 87 same-variable channels. This can be used as an `EARLY / TRANSITION / LATE` *representation* only if those names are explicitly defined as first-date / difference / second-date. It does **not** support claiming three real phenological stages or reproducing the paper's 58/62/55 mapping. The proposed `f1↔f88` pairing must not be used.
