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

**Location & Neighborhood**: narrowed from 18 to 7 via Pearson correlation, ANOVA F-tests, and a missingness/cardinality screen (drop if >80% missing or cardinality outside 2–5,000):

```python
location_features = [
    "Latitude",
    "Longitude",
    "City",
    "PostalCode",
    "CountyOrParish",
    "MLSAreaMajor",
    "HighSchoolDistrict",
]
```

## Preprocessing Steps

1. **Target transformation:** `log_price = log(ClosePrice)`, for `ClosePrice > 0`. Training on the log scale (rather than raw dollars) was validated via an A/B test in `05_advanced_models.ipynb` — the log-target model outperformed a raw-dollar-target model on every metric (RMSE, MAE, R², MAPE).
2. **Geographic cleaning:** Dropped rows with coordinates outside California's valid range; fixed 12 longitude sign-flip errors (positive instead of negative longitude) rather than discarding them; 37 remaining invalid-coordinate rows dropped.
3. **Feature engineering:** `bed_bath_ratio`, `property_age` (negative ages from data-entry errors clipped to 0), `amenity_score`.
4. **School district spatial join:** `geopandas` point-in-polygon join against a California school-district boundary file to assign `DistrictName`; unmatched properties (24.19%) flagged as missing rather than dropped.
5. **Missing value handling:** Median/mode imputation with a `_missing` indicator flag per column, so the model can still learn from the fact that a value was missing.
6. **Data-quality filtering (two layers, target-level):**
   - **ClosePrice floor/cap:** computed via IQR on `log_price` (rather than raw `ClosePrice`, which is heavily right-skewed) and converted back to dollar scale. This removes implausible entries at both ends — e.g. sales recorded under $100, and sales in the hundreds of millions with a price-per-square-foot far outside anything seen in legitimate luxury listings. Verified against actual rows on both sides of the cutoff to confirm real high-end sales (e.g. $16–16.5M homes with proportional square footage) are not excluded.
   - **`price_ratio` (ClosePrice / ListPrice) filtering:** capped at the train-set 0.5th–99.95th percentiles, to catch a different failure mode — a plausible `ClosePrice` that is wildly mismatched against its own `ListPrice` (bad `ListPrice`, or a mismatched record). Overlap with the floor/cap filter above is under 5%, confirming this is an additive check, not a duplicate.

   *Note: this two-layer filter was motivated by diagnosing a severe dollar-scale R² collapse in the Linear Regression baseline (see `03_baseline_model.ipynb`). Tree-based models (Decision Tree, Random Forest, XGBoost, LightGBM) do not extrapolate on extreme x-values the way linear models do, so they were largely unaffected by the feature-level issues — but all models still benefit from removing target-level (`ClosePrice`) data errors, since bad labels teach any model the wrong signal regardless of architecture.*
7. **Categorical encoding:** High-cardinality fields (City, PostalCode, CountyOrParish, MLSAreaMajor, HighSchoolDistrict, DistrictName) use smoothed target encoding fit on train only; low-cardinality YN flags use one-hot encoding.
8. **Normalization:** `StandardScaler`, applied only where needed (Linear Regression), not for tree-based models.
9. **Final feature set:** 42 modeling features exported to `train.csv` / `validation.csv` / `test.csv`.

## Models Tested

| Model | Tuning approach |
|---|---|
| Linear Regression | VIF-based feature pruning (dropped all perfectly-collinear columns); IQR-based outlier caps on extreme numeric features; evaluated on the dollar scale after applying the data-quality caps above |
| Decision Tree | `GridSearchCV` with `TimeSeriesSplit` CV and a custom dollar-scale R² scorer |
| Random Forest | `GridSearchCV` with `TimeSeriesSplit` CV and a custom dollar-scale R² scorer |
| XGBoost | Grid search (max_depth, learning_rate, n_estimators, R²-scored) → luxury sample weighting → blend ratio search → regularization search |
| LightGBM | Same pipeline as XGBoost |

**Additional experiments — weighting and blending:**

- **Luxury sample weighting:** properties over $2M are underrepresented in the training data and consistently showed higher error. Weight multipliers from 1× to 10× were swept independently for each model, scored by validation R². XGBoost and LightGBM did not peak at the same multiplier — **XGBoost: 4× (val R² 0.8681), LightGBM: 2× (val R² 0.8628)**.
- **Blend ratio search:** with both base models weighted, the XGBoost/LightGBM blend ratio was swept from 0 to 1 in 0.1 increments, scored by validation R². Optimal split: **70% XGBoost / 30% LightGBM (val R² 0.8699)**.
- **Regularization search** (`RandomizedSearchCV`, 50 candidates × 5-fold CV) to reduce train-validation overfitting — this reduced the train-validation gap but did **not** improve test-set performance over the unregularized weighted ensemble (see comparison table below), so it was not used in the final model.
- **Luxury-segment specialist model** (trained only on >$2M properties) was tested and **rejected** — it underperformed the global model on the same subset, confirming the luxury segment's sample size is too small to support a standalone model.

**Model comparison (test set):**

| Model | Train R² | Val R² | Test R² | MAPE | MdAPE | Train-Val Gap |
|---|---|---|---|---|---|---|
| XGBoost (baseline, tuned) | 0.9479 | 0.8629 | 0.8426 | 11.49% | 7.92% | 0.0850 |
| LightGBM (baseline, tuned) | 0.9398 | 0.8602 | 0.8484 | 11.56% | 7.96% | 0.0796 |
| XGBoost weighted (no reg) | 0.9635 | 0.8681 | 0.8373 | 12.06% | 8.25% | 0.0954 |
| LightGBM weighted (no reg) | 0.9497 | 0.8628 | 0.8482 | 11.77% | 8.08% | 0.0869 |
| **Ensemble (weighted, no reg) — final model** | **0.9589** | **0.8695** | **0.8463** | **11.76%** | **8.07%** | **0.0894** |
| XGBoost auto-tuned (weighted + regularized) | 0.9043 | 0.8571 | 0.8367 | 13.05% | 8.92% | 0.0472 |
| LightGBM auto-tuned (weighted + regularized) | 0.9178 | 0.8573 | 0.8422 | 12.77% | 8.66% | 0.0605 |
| Ensemble (auto-tuned, regularized) | 0.9130 | 0.8597 | 0.8415 | 12.79% | 8.72% | 0.0533 |

*Note: the regularized/auto-tuned ensemble was an earlier candidate for "final model" — it succeeds at shrinking the train-validation gap, but scores worse than the unregularized weighted ensemble on every test-set metric. The unregularized version is therefore the model reported as final throughout this document.*

## Best Results

**Final model:** Weighted ensemble of XGBoost and LightGBM (70% XGBoost / 30% LightGBM), each individually weighted toward luxury (>$2M) listings before blending (XGBoost 4×, LightGBM 2×). Selected over the individually-best single model (LightGBM weighted, test R² 0.8482) because it better balances overall accuracy with accuracy specifically on the highest-value, highest-risk segment — see price-band breakdown below.

**Test set performance:**

| Metric | Value |
|---|---|
| Test R² | 0.8463 |
| MAPE | 11.76% |
| MdAPE | 8.07% |
| Train-Val Gap | 0.0894 |

**Performance by price band (test set) — why the ensemble, not the LightGBM baseline:**

| Price Band | LightGBM (baseline) MAPE | Ensemble (weighted) MAPE | LightGBM (baseline) MdAPE | Ensemble (weighted) MdAPE |
|---|---|---|---|---|
| <$500K | 15.27% | 15.48% | 8.45% | 8.37% |
| $500K–$1M | 8.88% | 9.00% | 6.20% | 6.28% |
| $1M–$2M | 11.40% | 12.22% | 8.89% | 9.32% |
| $2M+ | 16.20% | 14.98% | 13.22% | 11.74% |

Overall, the LightGBM baseline and the weighted ensemble score nearly identically (test R² 0.8484 vs 0.8463) — but broken out by price band, the ensemble trades a small amount of accuracy in the $1M–$2M range (MAPE +0.82pt) for a meaningfully more reliable estimate on $2M+ listings (MAPE −1.22pt, MdAPE −1.49pt). Given that mispricing a luxury property carries disproportionate business risk, this trade-off was judged worthwhile.

LightGBM weighted alone (test R² 0.8482, the highest of any single model) was also considered instead of the blend — but its own $2M+ MAPE only improves to 15.28%, well short of the ensemble's 14.98%. XGBoost weighted alone reaches the best $2M+ MAPE (14.53%) but at the cost of the lowest overall test R² of the five weighted/baseline candidates (0.8373). The blend is the only candidate that captures nearly all of XGBoost's luxury-segment gain while keeping nearly all of LightGBM's overall accuracy.

## Repository / Notebook Order

Run in this order — each stage depends on outputs from the previous one:

1. `01_exploration.ipynb` — EDA
2. `02_preprocessing.ipynb` — cleaning, feature engineering, encoding, train/val/test export
3. `03_baseline_model.ipynb` — Linear Regression
4. `04_model_comparison.ipynb` — Decision Tree, Random Forest
5. `05_advanced_models.ipynb` — XGBoost, LightGBM, weighting, blending, regularization, final evaluation and price-band analysis
