```markdown
# idx-exchange: California Property Close Price Prediction

Predicting the close (final sale) price of California single-family residential
properties using historical CRMLS (California Regional Multiple Listing Service)
data.

## Project Background

This project is part of a 12-week data science internship. The goal is to build a
machine learning model that predicts the close price of any single-family
residential property in California (currently for sale or not) based on its
characteristics at the time of the query.

`ClosePrice` is the target variable. Input features are drawn from the
property's characteristics — living area, bedrooms, bathrooms, lot size,
location, and more — sourced from CRMLS.

## Data

- **Source**: CRMLS (`CRMLSSold*.csv` files), accessed via FTP from
  `raw/California`.
- **Scope**: `PropertyType = Residential` and
  `PropertySubType = SingleFamilyResidence` only, per task requirements.
- **Coverage used in this analysis**: `california_sfr_202505_202605.csv`
  (~142K rows, 12 months of sales).
- **Metadata**: field definitions are documented in
  `Trestle Property MetaData.pdf` (see `resources/`).

Raw data files are not committed to this repo — see `data_setup.ipynb` for how
to load them locally.

## Repo Structure

```
idx-exchange/
├── README.md
├── data_setup.ipynb        # Data loading utilities
├── 01_exploration.ipynb    # Week 2: EDA, data cleaning, feature selection
└── .gitignore
```

## Progress

### ✅ Week 1 — Orientation & Setup
- Confirmed dataset access via FTP.
- Reviewed `Trestle Property MetaData.pdf` for field definitions.

### ✅ Week 2 — Data Exploration (`01_exploration.ipynb`)

**Target variable (`ClosePrice`) cleaning**
- Found and removed 1 record with `ClosePrice <= 0` — this had been silently
  corrupting all downstream correlation/ANOVA calculations (`log(0) = -inf`,
  which `dropna()` does not catch).
- Computed `price_ratio = ClosePrice / ListPrice` to catch systematic data
  entry errors. Found a batch of records with `price_ratio` ≈ 1000 — a
  confirmed unit error, corrected by dividing by 1000 rather than dropping.
  Remaining extreme ratios (~0.3–0.5% of rows) were filtered using
  data-driven quantile cutoffs (0.5th–99.95th percentile).
- Modeling target: `log_price = log(ClosePrice)`.

**Location & Neighborhood feature selection**
- Evaluated 18 location-related columns using Pearson correlation (numeric)
  and ANOVA F-tests (categorical) against `log_price`.
- **Retained 7 features**: `Latitude`, `Longitude`, `City`, `PostalCode`,
  `CountyOrParish`, `MLSAreaMajor`, `HighSchoolDistrict`.
- Dropped columns with 100% missingness, ID-like cardinality
  (`UnparsedAddress`), no variance (`StateOrProvince`), or excessive missing
  rates relative to explanatory power (`HighSchool`, `MiddleOrJuniorSchool`,
  `ElementarySchool`).

**EDA on retained features**
1. **Geographic distribution** — found and corrected longitude sign-flip
   errors (12 rows); dropped remaining unrecoverable out-of-state/invalid
   coordinates.
2. **Price distribution by county** — confirmed Bay Area counties (Santa
   Clara, San Mateo) have the highest median prices; inland counties (Kern,
   Merced) the lowest, consistent with known CA housing patterns.
3. **Missing value patterns** — `MLSAreaMajor` and `HighSchoolDistrict`
   missingness is not random; it's concentrated in specific counties,
   likely reflecting differences in regional MLS system coverage rather
   than data volume. Flagged as a model limitation.
4. **Sample size per category** — `PostalCode` and `City` have many sparse
   categories (<10 samples); will require smoothed target encoding or
   bucketing before use in non-tree models.
5. **Multicollinearity** — `City`, `PostalCode`, and `CountyOrParish` are
   highly overlapping (Cramér's V 0.91–0.98); `MLSAreaMajor` is the most
   independent location signal. Relevant for Linear Regression (coefficient
   stability) but not for tree-based models.

### ⬜ Week 3 — Data Preprocessing
- Handle missing values (impute/flag), encode categorical variables, scale
  numeric features.
- Train/test split: most recent month as test set, preceding X months as
  training (X to be tuned).

### ⬜ Week 4 — Baseline Model (Linear Regression)
### ⬜ Week 5 — Additional Models (Decision Tree, Random Forest)
### ⬜ Week 6 — Feature Engineering (bed/bath ratio, property age, school
district spatial join)
### ⬜ Week 7 — Advanced Models (XGBoost / LightGBM)
### ⬜ Week 8 — Evaluation Expansion (MAPE, MdAPE)
### ⬜ Week 9 — OPTIONAL: Streamlit Prediction App
### ⬜ Week 10 — Documentation
### ⬜ Week 11 — Practice Presentation
### ⬜ Week 12 — Final Presentation & Handoff

## Methodology Notes

- **Target variable**: `log(ClosePrice)`, chosen for its closer-to-normal
  distribution and better suitability for regression.
- **Train/test split**: time-based (train on earlier months, test on the
  most recent month) to reflect real-world deployment — the model must
  predict *future* prices, not interpolate within the same time period.
- **Model-specific feature handling**: tree-based models (Decision Tree,
  Random Forest, LightGBM) can consume raw categorical columns and missing
  values natively; Linear Regression requires encoding, imputation, and
  more careful handling of multicollinearity.
