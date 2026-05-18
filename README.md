# Online Shoppers Revenue Prediction

Applied Machine Learning project: predict whether an online shopping session will generate revenue.

## Dataset

- File: `online_shoppers_intention.csv`
- Source: UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset
- Target: `Revenue` (True / False)
- Size: 12,330 sessions, 17 explanatory variables, 0 missing values

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run exploratory data analysis:

```bash
python src/eda.py
```

Run the full modelling pipeline:

```bash
python src/train_models.py
```

Both scripts are fully self-contained. After running, all outputs are automatically placed in the correct canonical folders — no manual file moving is needed.

## Models

- **Champion**: Random Forest with `balanced_subsample` class weighting
- **Challengers**: Gradient Boosting, Decision Tree, Logistic Regression, KNN
- **Baseline**: DummyClassifier (most-frequent strategy)

The primary evaluation metric is F1-score for the positive class (`Revenue=True`) because the target is imbalanced (~15.5% positive rate). PR-AUC and ROC-AUC are also reported.

Two feature scenarios are evaluated:

- `with_page_values`: full model using all 17 variables.
- `without_page_values`: model with `PageValues` removed, testing robustness for early-session scoring when this feature is not yet available.

## Output structure

```
outputs/
  main/                    ← primary tables for the report
  appendix/
    eda/                   ← full EDA tables
    model/                 ← detailed model results, feature importance, classification reports
    thresholds/            ← full threshold sweep grids

reports/
  report.md                ← main report
  figures/
    main/                  ← figures embedded in the report body
    appendix/
      eda/                 ← secondary EDA figures
      model/               ← secondary model figures (ROC, PR, confusion matrix, SHAP, learning curves)
```

## Key outputs (after running)

| File | Content |
| --- | --- |
| `outputs/main/01_dataset_profile.csv` | Dataset size, missing values, duplicates, positive rate |
| `outputs/main/03_model_results_by_scenario.csv` | All model metrics for both scenarios |
| `outputs/main/06_threshold_summary_by_scenario.csv` | Default, F1-optimal, and business-optimal thresholds |
| `outputs/main/07_business_cost_matrix.csv` | TP/FP/FN/TN cost assumptions |
| `outputs/main/08_business_threshold_summary.csv` | Business value at key thresholds |
| `reports/figures/main/` | All figures used in the report |
| `reports/figures/appendix/model/A17_champion_shap.png` | SHAP beeswarm for the champion |
| `reports/figures/appendix/model/A15_learning_curves_with_page_values.png` | Learning curves |
| `reports/figures/main/10_calibration_curves.png` | Reliability diagrams |

## Business evaluation

The champion is evaluated with a custom cost matrix (TP = +20, FP = −2, FN = 0, TN = 0) and a full threshold sweep from 0.05 to 0.95. The business-optimal threshold (maximising expected value on the test set) is reported alongside the F1-optimal threshold.

## Sources

- Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

