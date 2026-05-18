# Outputs

Use `outputs/main/` for the report and presentation. It contains only the tables needed to support the main story.

## Main Outputs

| File | Content |
| --- | --- |
| `main/01_dataset_profile.csv` | Dataset size, missing values, duplicates and target rate |
| `main/02_target_distribution.csv` | Revenue class distribution |
| `main/03_model_results_by_scenario.csv` | All model metrics for both feature scenarios |
| `main/04_champion_results_by_scenario.csv` | Champion-only comparison with and without `PageValues` |
| `main/05_page_values_impact.csv` | Metric differences caused by removing `PageValues` |
| `main/06_threshold_summary_by_scenario.csv` | Default, F1-optimal, balanced-accuracy and business thresholds |
| `main/07_business_cost_matrix.csv` | Business cost/gain assumptions |
| `main/08_business_threshold_summary.csv` | Business value for key thresholds |

## Appendix Outputs

Use `outputs/appendix/` only if more detail is needed.

- `appendix/eda/`: full exploratory data analysis tables (A1–A10).
- `appendix/model/`: detailed model results, feature importance, CV scores, and classification reports (A11–A24).
- `appendix/thresholds/`: full threshold search grids (A19–A22).

## How outputs are generated

All outputs are created automatically by running the scripts:

```bash
python src/eda.py        # generates EDA tables → outputs/main/ and outputs/appendix/eda/
python src/train_models.py  # generates model tables → outputs/main/ and outputs/appendix/
```

No manual file moving is required. Running the scripts again will overwrite all canonical outputs with fresh results.

