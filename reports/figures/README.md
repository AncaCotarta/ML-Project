# Figures

Use `main/` for figures embedded in the report body. Use `appendix/` for secondary figures.

## Main figures

| File | Content |
| --- | --- |
| `main/01_target_distribution.png` | Target class imbalance |
| `main/02_conversion_by_month.png` | Conversion rate by month |
| `main/03_page_values_by_revenue.png` | PageValues distribution by revenue outcome |
| `main/04_champion_page_values_impact.png` | Champion performance with vs without PageValues |
| `main/05_model_f1_comparison.png` | F1-score comparison across all models |
| `main/06_champion_confusion_matrix.png` | Champion confusion matrix |
| `main/07_threshold_tuning_with_page_values.png` | Threshold tuning curve |
| `main/08_business_value_by_threshold.png` | Expected business value by threshold |
| `main/09_roc_pr_comparison_all_models.png` | ROC and PR curves for all models |
| `main/10_calibration_curves.png` | Reliability diagrams (probability calibration) |
| `main/11_cv_scores_summary.png` | Cross-validation F1 scores with standard deviation |

## Appendix figures

- `appendix/eda/`: EDA figures (A1–A5)
- `appendix/model/`: Model figures (A6–A19) including SHAP, learning curves, feature importance

## How figures are generated

All figures are created automatically by running the scripts:

```bash
python src/eda.py        # generates EDA figures → reports/figures/main/ and appendix/eda/
python src/train_models.py  # generates model figures → reports/figures/main/ and appendix/model/
```

No manual file moving is required.

