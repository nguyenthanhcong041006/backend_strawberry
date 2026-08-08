# EDA Plan

EDA provides the evidence base for the paper and helps the team detect data problems before model development.

Owner: Hai  
Reviewers: Gate checker, with Hung for data-collection assumptions and Cong for preprocessing assumptions

## Objectives

- Describe the dataset clearly.
- Detect missing, invalid, or inconsistent data.
- Show temporal changes in visual and numeric features.
- Show spatial/ROI variation across fruit positions.
- Produce paper-ready figures and tables.
- Feed quality findings back into preprocessing and labeling.

## Required Reports

Store reports under:

```text
output/reports/eda/
```

Minimum reports:

- dataset inventory by experiment, date, fruit type, and fruit ID;
- raw vs processed vs excluded frame counts;
- missing-value report for temperature, humidity, and firmness;
- timestamp coverage and capture interval consistency;
- EOL/RUL distribution after labeling;
- fruit-level sequence length summary;
- sensor range and anomaly report;
- preprocessing QC summary from exclusion and mask reports.

## Required Graphs

Store graphs under:

```text
output/graphs/eda/
```

Minimum graphs:

- frame count by date and fruit ID;
- RUL distribution by fruit ID;
- temperature and humidity over time;
- firmness trend over time for avocado;
- color-channel or CIELAB trend over time;
- example temporal image strips per fruit;
- spatial/ROI comparison across the 3x2 layout;
- excluded-frame reason distribution;
- mask/foreground area trend over time where masks are available.

## Temporal Analysis

Temporal analysis should answer:

- How does each fruit visually change over time?
- Does RUL decrease smoothly after labeling?
- Are there capture gaps or sensor gaps?
- Are daily firmness values consistent with visual decay for avocado?
- Are disturbance periods visible around measurement times?

## Spatial Analysis

Spatial analysis should answer:

- Do ROI positions differ in lighting or visibility?
- Do some positions have more failed masks?
- Are color distributions biased by position?
- Are environmental values shared correctly across fruit IDs?
- Does any fruit have unusually short or long usable life?

## Acceptance Criteria

EDA is ready for paper support when:

- each figure has a clear source file and generation script/notebook;
- each reported count can be traced to a manifest;
- missing values are explained, not ignored;
- dataset limitations are explicitly listed;
- plots distinguish strawberry, avocado, and legacy datasets where applicable;
- no EDA output mixes legacy strawberry data with current data without labeling it.

