# How to run the pipeline

The full pipeline (`src/train_models.py`) trains 6 models with GridSearchCV, produces
learning curves, SHAP values, calibration curves, and ~30 figures.
Expected wall-clock time depends on hardware:

| Hardware | Estimated time |
|---|---|
| Colab Pro (2–4 CPU cores, n_jobs=-1) | 15–25 min |
| Modern laptop, 6–8 cores (n_jobs=-1) | 8–15 min |
| Single-core / low-RAM machine | 45–90 min |

Note: scikit-learn does **not** use GPU. GPU hardware gives no speed benefit here.
The speedup comes from multi-core CPU parallelism, which is already enabled
(`n_jobs=-1` on GridSearchCV, RandomForest, permutation importance, and learning curves).

---

## Option A — Local laptop (recommended if 6+ cores available)

```bash
# 1. Navigate to the project folder
cd "path/to/Project ML"

# 2. Install dependencies (once)
pip install -r requirements.txt

# 3. Run EDA (fast, ~10 seconds)
python src/eda.py

# 4. Run the full modelling pipeline
python src/train_models.py
```

All outputs are written automatically to `outputs/` and `reports/figures/`.
No manual file moving is needed.

---

## Option B — Google Colab Pro

### Step 1 — upload the project

Either mount Google Drive and copy the folder there, or upload files directly.
The minimum required files are:

```
online_shoppers_intention.csv
src/train_models.py
src/eda.py
requirements.txt
```

The output folders are created automatically by the script.

### Step 2 — install dependencies

In a Colab cell:

```python
!pip install -q pandas numpy scikit-learn matplotlib shap
```

### Step 3 — run EDA (optional but fast)

```python
!python src/eda.py
```

### Step 4 — run the full pipeline

```python
!python src/train_models.py
```

Expected output: a printed table of all model results at the end.

### Step 5 — download outputs

After the run finishes, zip and download the outputs:

```python
import shutil
shutil.make_archive("project_outputs", "zip", ".", "outputs")
shutil.make_archive("project_figures", "zip", ".", "reports/figures")
```

Then download `project_outputs.zip` and `project_figures.zip` from the Colab
file browser (left panel → right-click → Download).

Unzip both into the same `Project ML/` folder on your computer, replacing the
existing `outputs/` and `reports/figures/` directories. The report (`reports/report.md`)
will then have all figures and tables resolved correctly.

---

## What the pipeline produces

| Output | Location |
|---|---|
| All model metrics (both scenarios) | `outputs/main/03_model_results_by_scenario.csv` |
| Champion vs. challengers comparison | `outputs/main/04_champion_results_by_scenario.csv` |
| Threshold sweep results | `outputs/appendix/thresholds/` |
| Classification reports (text) | `outputs/appendix/model/A18_classification_reports.txt` |
| Main report figures (embedded in report.md) | `reports/figures/main/` |
| Appendix figures | `reports/figures/appendix/` |

After the run, send the updated `outputs/main/03_model_results_by_scenario.csv`
so the report can be finalised with the real Gradient Boosting numbers.

---

## Verification after the run

Check that these files exist — they are the ones that were previously missing:

```
reports/figures/main/09_roc_pr_comparison_all_models.png
reports/figures/main/10_calibration_curves.png
reports/figures/main/11_cv_scores_summary.png
reports/figures/appendix/model/A15_learning_curves_with_page_values.png
reports/figures/appendix/model/A17_champion_shap.png
outputs/appendix/model/A18_classification_reports.txt   (Gradient Boosting section should be non-empty)
```

Quick check command (run in the project folder):

```bash
python - <<'EOF'
from pathlib import Path
must_exist = [
    "reports/figures/main/09_roc_pr_comparison_all_models.png",
    "reports/figures/main/10_calibration_curves.png",
    "reports/figures/main/11_cv_scores_summary.png",
    "reports/figures/appendix/model/A15_learning_curves_with_page_values.png",
    "reports/figures/appendix/model/A17_champion_shap.png",
]
for p in must_exist:
    status = "OK" if Path(p).exists() else "MISSING"
    print(f"[{status}] {p}")

import pandas as pd
df = pd.read_csv("outputs/main/03_model_results_by_scenario.csv")
gb = df[df["model"] == "gradient_boosting"]
if gb.empty:
    print("[MISSING] Gradient Boosting results in CSV")
else:
    print(f"[OK] Gradient Boosting found — with_page_values F1 = {gb[gb['scenario']=='with_page_values']['f1'].values[0]:.4f}")
EOF
```

Once you confirm everything is green, share the terminal output of that check
and the `outputs/main/03_model_results_by_scenario.csv` file. The report will
then be updated with the real numbers and finalised.
