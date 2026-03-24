## Comprehensive Data Imputation and Correction Strategy

### 1. Standard Gap Imputation (Missing/Invalid Data)
| Category | Method Tag | Description / Logic |
| :--- | :--- | :--- |
| **≤ 3 Hours** | `LINEAR` | Linear interpolation between pre- and post-gap values. |
| **≤ 1 Week** | `WEEK_BEFORE` | Imputation using values from the exact same hour of the previous week. |
| **> 1 Week** | `ZERO` | Default fallback for extended missing data blocks where temporal interpolation is unfeasible. Assumes component was physically offline. |
| **Edge Cases** | `FORWARD_FILL` | Last valid observation carried forward (for gaps extending to the end of the series). |
| **Edge Cases** | `BACKWARD_FILL` | Next valid observation carried backward (for gaps at the start of the series). |
| **Outliers** | `FILTERED_OUTLIER_*` | Prefix added to standard methods (e.g., `LINEAR`) if the gap was caused by clipping extreme, unrealistic values (e.g., values > 100,000). |
| **Cross-domain** | `DAYAHEAD_PROXY` | (Intraday Commercial Flows Only) Long gaps patched using day-ahead schedule data. |

### 2. System Zero Corrections (Catastrophic Drops)
| Category | Method Tag | Description / Logic |
| :--- | :--- | :--- |
| **Gen. Zero** | `GEN_ZERO_*` | Systemic 0 MW total generation detected. Patched via tiered logic (`LINEAR`, `WEEK_BEFORE`, or `LONG_GAP_GLOBAL_MEAN`). |
| **Load Zero** | `LOAD_ZERO_*` | Systemic 0 MW total load detected. Patched via tiered logic. |
| **Flow System Zero**| `PHYS_FLOW_SYSTEM_ZERO_*` | Total API failure where all physical borders report 0 MW. Patched via tiered logic. |
| **Flow AC Zero** | `PHYS_BILATERAL_ZERO_[X]`| Bilateral failure where both directions of an AC line to country [X] report exactly 0 MW. Patched via tiered logic. |
| **Bypass** | `VALID_ZERO_ISLAND_IGNORED`| Intentionally skipped patching because the 0 MW cross-border flow occurred in a configured "True Island" (e.g., Sardinia) where such states are physically valid. |
| **Data Cleaning** | `CLIPPED_NEGATIVE` | Negative generation values corrected to 0 MW. |

### 3. Network Symmetry Balancing
| Category | Method Tag | Description / Logic |
| :--- | :--- | :--- |
| **Tie-Breaker** | `SYMMETRY_TRUSTED_TIEBREAKER_[X]` | Bilateral conflict where both zones report differing data (due to a different time of download or gap-filling), but zone [X] is deemed more reliable (already filled with non-zeros) and overwrites the target zone. |
| **Missing Data** | `SYMMETRY_PRIORITY_NON_ZERO_[X]` | [X], the zone which has been gap-filled with non-zero values overwrites Target zone's data to restore physical symmetry. |
