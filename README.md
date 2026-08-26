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

## Dataset Source

- **Source:** CRMLS (California Regional Multiple Listing Service) monthly sold-listing CSV exports
- **Scope:** `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"` listings across California
- **Coverage:** 17 monthly files (Feb 2025 – Jun 2026); 186,196 rows after property-type filtering
- **Train / Validation / Test split** (time-based, by `CloseDate`):

| Split | Date range | Rows |
|---|---|---|
| Train | 2025-02-01 – 2026-01-31 | 129,477 |
| Validation | 2026-02-01 – 2026-05-31 | 43,755 |
| Test | 2026-06-01 – 2026-06-30 | 12,853 |


## Feature Selection Process

The team inventoried 83 raw CRMLS columns and divided them into 5 functional groups. Each member reviewed their assigned group; the team then collectively decided which features carried meaningful predictive signal.

| Group | Purpose | Raw columns | Final features |
|---|---|---|---|
| Property Features | Predictive features about the house itself | 26 | ~10 |
| Location & Neighborhood | Geographic and school-related features | 18 | 7 |
| Lot & Financial | Price, taxes, HOA, and lot characteristics | 13 | 4 |
| Listing & Transaction | Time-based listing history and sale information | 10 | 0 (excluded — mostly known only after sale closes) |
| Agents & Offices | Brokerage and agent metadata | 16 | 0 (excluded — not price-predictive) |

**Location & Neighborhood** : narrowed from 18 to 7 via Pearson correlation, ANOVA F-tests, and a missingness/cardinality screen (drop if >80% missing or cardinality outside 2–5,000):

\```python
location_features = [
    "Latitude",
    "Longitude",
    "City",
    "PostalCode",
    "CountyOrParish",
    "MLSAreaMajor",
    "HighSchoolDistrict",
]
\```

## Preprocessing Steps

1. **Target transformation:** `log_price = log(ClosePrice)`, for `ClosePrice > 0`. Training on the log scale (rather than raw dollars) was validated via an A/B test in `05_advanced_models.ipynb` — the log-target model outperformed a raw-dollar-target model on every metric (RMSE, MAE, R², MAPE)
2. **Geographic cleaning:** Dropped rows with coordinates outside California's valid range; fixed 12 longitude sign-flip errors (positive instead of negative longitude) rather than discarding them; 37 remaining invalid-coordinate rows dropped
3. **Feature engineering:** `bed_bath_ratio`, `property_age` (negative ages from data-entry errors clipped to 0), `amenity_score`
4. **School district spatial join:** `geopandas` point-in-polygon join against a California school-district boundary file to assign `DistrictName`; unmatched properties (24.19%) flagged as missing rather than dropped
5. **Missing value handling:** Median/mode imputation with a `_missing` indicator flag per column, so the model can still learn from the fact that a value was missing
6. **Outlier filtering:** Extreme `price_ratio` (ClosePrice / ListPrice) values — capped at the train-set 0.5th–99.95th percentiles — removed to eliminate likely data-entry errors. This had a very large impact on tree-based model performance (e.g., XGBoost validation R² improved from 0.02 to 0.84)
7. **Categorical encoding:** High-cardinality fields (City, PostalCode, CountyOrParish, MLSAreaMajor, HighSchoolDistrict, DistrictName) use smoothed target encoding fit on train only; low-cardinality YN flags use one-hot encoding
8. **Normalization:** `StandardScaler`, applied only where needed (Linear Regression), not for tree-based models
9. **Final feature set:** 42 modeling features exported to `train.csv` / `validation.csv` / `test.csv`


## Models Tested

| Model | Tuning approach |
|---|---|
| Linear Regression | VIF-based feature pruning; evaluated on the dollar scale after applying the data-quality caps above |
| Decision Tree | `GridSearchCV` with `TimeSeriesSplit` CV and a custom dollar-scale R² scorer |
| Random Forest | `GridSearchCV` with `TimeSeriesSplit` CV and a custom dollar-scale R² scorer |
| XGBoost | Grid search (max_depth, learning_rate, n_estimators) → sample weighting (3× for luxury properties) → `RandomizedSearchCV` for regularization → ensembling |
| LightGBM | Same pipeline as XGBoost |

**Additional experiments:**
- Sample weighting for luxury properties (>$2M) to address their underrepresentation in training data
- Weighted ensemble of XGBoost + LightGBM
- Regularization search (`RandomizedSearchCV`, 50 candidates × 5-fold CV) to reduce train-validation overfitting
- A luxury-segment specialist model (trained only on >$2M properties) was tested and **rejected** — it underperformed the global model on the same subset, confirming the luxury segment's sample size is too small to support a standalone model


**Model comparison (test set):**

**Model comparison (test set):**

| Model | R² | MAPE | MdAPE |
|---|---|---|---|
| Decision Tree | 0.7286 | 17.66% | 12.65% |
| Random Forest | 0.8177 | 11.98% | 7.83% |
| XGBoost | 0.8473 | 11.58% | 7.88% |
| LightGBM | 0.8480 | 11.65% | 8.02% |
| XGBoost weighted | 0.8493 | 11.91% | 8.13% |
| LightGBM weighted | 0.8573 | 11.90% | 8.13% |
| Ensemble (weighted, no reg) | 0.8575 | 11.77% | 8.01% |
| XGB auto-tuned (weighted + regularized) | 0.8497 | 12.89% | 8.86% |
| LGBM auto-tuned (weighted + regularized) | 0.8422 | 12.63% | 8.63% |
| Final Ensemble (auto-tuned) | 0.8502 | 12.71% | 8.67% |

## Best Results

**Final model:** Weighted ensemble of XGBoost (70%) and LightGBM (30%), each individually tuned via `RandomizedSearchCV` with regularization

**Test set performance:**

| Metric | Value |
|---|---|
| RMSE | $565,695 |
| MAE | $190,045 |
| R² | 0.8502 |
| MAPE | 12.71% |
| MdAPE | 8.67% |
| Train-Val Gap | 0.0608 |

**Performance by price band (test set, final ensemble):**

| Price Band | MAPE | MdAPE |
|---|---|---|
| <$500K | 16.94% | 9.18% |
| $500K–$1M | 9.84% | 6.84% |
| $1M–$2M | 13.44% | 10.33% |
| $2M+ | 15.46% | 12.21% |
