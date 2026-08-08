# Stage 1.5 Numeric Coverage Cross-Check

This final Stage 1.5 layer verifies coverage between the master image inventory and available numeric CSV files.

## Scope

- Environment readings are checked at timestamp level against webcam filename timestamps.
- Hardness readings are checked at date level because the current hardness CSV has daily fruit columns, not image-localized fruit IDs.
- This report verifies coverage only; it does not create labels or model-ready targets.

## Outputs

- Image coverage CSV: `D:\GW_UNIVERSITY\AIS\Fruit_shell_life\Env\UOG_AIS_Fruit_V2\Strawberry-RUL-prediction\data\02_processed\stage_1_5\sample\image_numeric_coverage.csv`
- Numeric coverage CSV: `D:\GW_UNIVERSITY\AIS\Fruit_shell_life\Env\UOG_AIS_Fruit_V2\Strawberry-RUL-prediction\data\02_processed\stage_1_5\sample\numeric_image_coverage.csv`

## Coverage Summary

| Metric | Count |
| --- | ---: |
| Inventory images checked | 100 |
| Images with parsed timestamp | 100 |
| Images without parsed timestamp | 0 |
| Images with hardness date record | 100 |
| Images missing hardness date record | 0 |
| Images with environment timestamp record | 100 |
| Images missing environment timestamp record | 0 |
| Numeric rows checked | 1038 |
| Numeric rows with matching image | 111 |
| Numeric rows missing matching image | 927 |

## Numeric Sources

| Source | Raw Rows | Coverage Rows |
| --- | ---: | ---: |
| hardness | 12 | 12 |
| environment | 1026 | 1026 |

## Warnings

No warnings.
