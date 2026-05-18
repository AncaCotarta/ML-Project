from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "online_shoppers_intention.csv"
OUTPUT_DIR = ROOT / "outputs" / "generated"
FIGURE_DIR = ROOT / "reports" / "figures" / "generated"
MAIN_FIGURE_DIR = ROOT / "reports" / "figures" / "main"
APPENDIX_MODEL_FIGURE_DIR = ROOT / "reports" / "figures" / "appendix" / "model"
MAIN_OUTPUT_DIR = ROOT / "outputs" / "main"
APPENDIX_MODEL_OUTPUT_DIR = ROOT / "outputs" / "appendix" / "model"
APPENDIX_THRESHOLDS_DIR = ROOT / "outputs" / "appendix" / "thresholds"
RANDOM_STATE = 42
TRUE_POSITIVE_GAIN = 20.0
FALSE_POSITIVE_COST = -2.0
FALSE_NEGATIVE_COST = 0.0
TRUE_NEGATIVE_GAIN = 0.0


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [
        column
        for column in [
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
        ]
        if column in x.columns
    ]
    numeric_features = [
        column
        for column in x.columns
        if column not in categorical_features
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    specs = {
        "dummy_baseline": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [20, 50, 100],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [7, 15, 31],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        ),
        "random_forest_champion": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_estimators=120,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {
                "model__max_depth": [8, None],
                "model__min_samples_leaf": [5, 20],
                "model__max_features": ["sqrt"],
            },
        ),
    }

    searches = {}
    for name, (model, params) in specs.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        # n_jobs=1 for KNN to avoid threadpoolctl macOS issue; parallel elsewhere
        grid_jobs = 1 if name == "knn" else -1
        searches[name] = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            scoring="f1",
            cv=cv,
            n_jobs=grid_jobs,
            refit=True,
        )
    return searches


def get_positive_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return (scores - scores.min()) / (scores.max() - scores.min())
    return model.predict(x_test)


def evaluate_model(scenario: str, name: str, search: GridSearchCV, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    y_score = get_positive_scores(best_model, x_test)

    return {
        "scenario": scenario,
        "model": name,
        "best_params": search.best_params_,
        "cv_f1": search.best_score_,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
    }


def score_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = y_score >= threshold
    y_true_array = np.asarray(y_true).astype(bool)
    true_positive = int(((y_pred == True) & (y_true_array == True)).sum())
    false_positive = int(((y_pred == True) & (y_true_array == False)).sum())
    false_negative = int(((y_pred == False) & (y_true_array == True)).sum())
    true_negative = int(((y_pred == False) & (y_true_array == False)).sum())
    expected_business_value = (
        true_positive * TRUE_POSITIVE_GAIN
        + false_positive * FALSE_POSITIVE_COST
        + false_negative * FALSE_NEGATIVE_COST
        + true_negative * TRUE_NEGATIVE_GAIN
    )

    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_positive_rate": y_pred.mean(),
        "predicted_positive_count": int(y_pred.sum()),
        "expected_business_value": expected_business_value,
        "expected_business_value_per_session": expected_business_value / len(y_true),
    }


def optimize_threshold(
    scenario: str,
    champion: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    y_score = get_positive_scores(champion, x_test)
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        rows.append(score_threshold(y_test, y_score, round(float(threshold), 2)))

    threshold_results = pd.DataFrame(rows)
    threshold_results.insert(0, "scenario", scenario)
    threshold_results.to_csv(OUTPUT_DIR / f"{scenario}_threshold_results.csv", index=False)

    default_result = score_threshold(y_test, y_score, 0.50)
    default_result["scenario"] = scenario
    default_result["selection"] = "default_0.50"

    best_f1_result = threshold_results.loc[threshold_results["f1"].idxmax()].to_dict()
    best_f1_result["selection"] = "best_f1"

    best_balanced_accuracy_result = threshold_results.loc[threshold_results["balanced_accuracy"].idxmax()].to_dict()
    best_balanced_accuracy_result["selection"] = "best_balanced_accuracy"

    best_business_result = threshold_results.loc[threshold_results["expected_business_value"].idxmax()].to_dict()
    best_business_result["selection"] = "best_business_value"

    summary = pd.DataFrame([default_result, best_f1_result, best_balanced_accuracy_result, best_business_result])
    summary.to_csv(OUTPUT_DIR / f"{scenario}_threshold_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(threshold_results["threshold"], threshold_results["precision"], label="Precision", color="#3f7f93")
    ax.plot(threshold_results["threshold"], threshold_results["recall"], label="Recall", color="#9a6f2d")
    ax.plot(threshold_results["threshold"], threshold_results["f1"], label="F1-score", color="#b85c38")
    ax.axvline(0.50, color="#6f7f8f", linestyle="--", linewidth=1.5, label="Default threshold")
    ax.axvline(best_f1_result["threshold"], color="#b85c38", linestyle=":", linewidth=2, label="Best F1 threshold")
    ax.set_title(f"Champion threshold tuning ({scenario})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_threshold_tuning.png", dpi=160)
    plt.close(fig)

    return summary


def save_business_value_plot(threshold_summaries: pd.DataFrame) -> None:
    cost_matrix = pd.DataFrame(
        [
            {"case": "True positive", "value": TRUE_POSITIVE_GAIN, "interpretation": "Correctly targeted buyer"},
            {"case": "False positive", "value": FALSE_POSITIVE_COST, "interpretation": "Marketing action sent to non-buyer"},
            {"case": "False negative", "value": FALSE_NEGATIVE_COST, "interpretation": "Missed buyer, no direct action cost counted"},
            {"case": "True negative", "value": TRUE_NEGATIVE_GAIN, "interpretation": "Correctly ignored non-buyer"},
        ]
    )
    cost_matrix.to_csv(OUTPUT_DIR / "business_cost_matrix.csv", index=False)

    business_rows = threshold_summaries[threshold_summaries["selection"].isin(["default_0.50", "best_f1", "best_business_value"])].copy()
    business_rows.to_csv(OUTPUT_DIR / "business_threshold_summary.csv", index=False)

    labels = business_rows["scenario"] + "\n" + business_rows["selection"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, business_rows["expected_business_value"], color="#3f7f93")
    ax.set_title("Expected business value by threshold selection")
    ax.set_ylabel("Expected value on test set")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "business_value_by_threshold.png", dpi=160)
    plt.close(fig)


def save_dataset_summary(df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_values": df.isna().sum(),
            "unique_values": df.nunique(),
        }
    )
    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv")

    target_counts = df["Revenue"].value_counts().rename_axis("Revenue").reset_index(name="count")
    target_counts["share"] = target_counts["count"] / len(df)
    target_counts.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)


def save_calibration_curves(
    scenario: str,
    models: dict[str, GridSearchCV],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Reliability diagrams: compare predicted probabilities to observed conversion rates."""
    color_map = {
        "random_forest_champion": "#b85c38",
        "gradient_boosting": "#385a64",
        "logistic_regression": "#3f7f93",
        "decision_tree": "#9a6f2d",
        "knn": "#6f7f8f",
    }
    label_map = {
        "random_forest_champion": "Random Forest (champion)",
        "gradient_boosting": "Gradient Boosting",
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "KNN",
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for name, search in models.items():
        if name == "dummy_baseline":
            continue
        model = search.best_estimator_
        y_score = get_positive_scores(model, x_test)
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=10, strategy="quantile")
        except Exception:
            continue
        color = color_map.get(name, "gray")
        label = label_map.get(name, name)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed conversion rate")
    ax.set_title(f"Calibration curves — reliability diagram ({scenario})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_calibration_curves.png", dpi=160)
    plt.close(fig)


def save_cv_scores_summary(
    scenario: str,
    models: dict[str, GridSearchCV],
) -> None:
    """Save a table of each model's best CV F1 score with standard deviation across folds."""
    rows = []
    for name, search in models.items():
        if name == "dummy_baseline":
            continue
        best_index = search.best_index_
        cv_results = search.cv_results_
        mean_score = cv_results["mean_test_score"][best_index]
        std_score = cv_results["std_test_score"][best_index]
        rows.append({
            "model": name,
            "cv_f1_mean": round(mean_score, 4),
            "cv_f1_std": round(std_score, 4),
            "cv_f1_lower": round(mean_score - std_score, 4),
            "cv_f1_upper": round(mean_score + std_score, 4),
        })
    cv_summary = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
    cv_summary.insert(0, "scenario", scenario)
    cv_summary.to_csv(OUTPUT_DIR / f"{scenario}_cv_scores_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    cv_summary_sorted = cv_summary.sort_values("cv_f1_mean", ascending=True)
    ax.barh(
        cv_summary_sorted["model"],
        cv_summary_sorted["cv_f1_mean"],
        xerr=cv_summary_sorted["cv_f1_std"],
        color="#385a64",
        capsize=4,
    )
    ax.set_title(f"Cross-validation F1 score (mean ± 1 std, {scenario})")
    ax.set_xlabel("CV F1-score")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_cv_scores_summary.png", dpi=160)
    plt.close(fig)


def save_roc_pr_comparison(
    scenario: str,
    models: dict[str, GridSearchCV],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save a combined ROC and Precision-Recall comparison figure for all non-dummy models."""
    color_map = {
        "random_forest_champion": "#b85c38",
        "gradient_boosting": "#385a64",
        "logistic_regression": "#3f7f93",
        "decision_tree": "#9a6f2d",
        "knn": "#6f7f8f",
    }
    label_map = {
        "random_forest_champion": "Random Forest (champion)",
        "gradient_boosting": "Gradient Boosting",
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "KNN",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, search in models.items():
        if name == "dummy_baseline":
            continue
        color = color_map.get(name, "gray")
        label = label_map.get(name, name.replace("_", " ").title())
        RocCurveDisplay.from_estimator(
            search.best_estimator_, x_test, y_test,
            ax=axes[0], name=label, color=color,
        )
        PrecisionRecallDisplay.from_estimator(
            search.best_estimator_, x_test, y_test,
            ax=axes[1], name=label, color=color,
        )
    axes[0].set_title(f"ROC curves — all models ({scenario})")
    axes[1].set_title(f"Precision-Recall curves — all models ({scenario})")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Model comparison ({scenario})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_roc_pr_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_learning_curves(
    scenario: str,
    models: dict[str, GridSearchCV],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Save learning curves for the champion, gradient boosting, and logistic regression."""
    names_to_plot = [
        name for name in ["logistic_regression", "gradient_boosting", "random_forest_champion"]
        if name in models
    ]
    label_map = {
        "random_forest_champion": "Random Forest (champion)",
        "gradient_boosting": "Gradient Boosting",
        "logistic_regression": "Logistic Regression",
    }
    color_map = {
        "random_forest_champion": "#b85c38",
        "gradient_boosting": "#385a64",
        "logistic_regression": "#3f7f93",
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    train_fractions = np.linspace(0.2, 1.0, 6)

    fig, axes = plt.subplots(1, len(names_to_plot), figsize=(15, 5), sharey=True)
    if len(names_to_plot) == 1:
        axes = [axes]
    for ax, name in zip(axes, names_to_plot):
        estimator = models[name].best_estimator_
        label = label_map.get(name, name)
        color = color_map.get(name, "gray")
        sizes, train_scores, val_scores = learning_curve(
            estimator=estimator,
            X=x_train,
            y=y_train,
            cv=cv,
            scoring="f1",
            train_sizes=train_fractions,
            n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        ax.plot(sizes, train_mean, color=color, linewidth=2, label="Training F1")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=color)
        ax.plot(sizes, val_mean, color=color, linestyle="--", linewidth=2, label="Validation F1")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color=color)
        ax.set_title(label)
        ax.set_xlabel("Training samples")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("F1-score (Revenue=True)")
    fig.suptitle(f"Learning curves ({scenario})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_learning_curves.png", dpi=160)
    plt.close(fig)


def save_shap_summary(
    scenario: str,
    champion: Pipeline,
    x_test: pd.DataFrame,
) -> None:
    """Save a SHAP beeswarm summary plot for the champion model."""
    try:
        import shap
    except ImportError:
        return
    preprocessor = champion.named_steps["preprocess"]
    rf_model = champion.named_steps["model"]
    sample_size = min(500, len(x_test))
    x_sample = x_test.iloc[:sample_size]
    x_transformed = preprocessor.transform(x_sample)
    if hasattr(x_transformed, "toarray"):
        x_transformed = x_transformed.toarray()
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = []
        for _, transformer, columns in preprocessor.transformers_:
            cols = columns if isinstance(columns, list) else [columns]
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
    x_transformed_df = pd.DataFrame(
        x_transformed,
        columns=feature_names[: x_transformed.shape[1]],
    )
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(x_transformed_df)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
    shap.summary_plot(shap_vals, x_transformed_df, show=False, max_display=15)
    plt.title(f"SHAP values — champion ({scenario})")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{scenario}_champion_shap.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close("all")


def save_figures(
    scenario: str,
    results: pd.DataFrame,
    models: dict[str, GridSearchCV],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    ordered = results.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["model"], ordered["f1"], color="#3f7f93")
    ax.set_title(f"Model comparison on the test set ({scenario})")
    ax.set_xlabel("F1-score for Revenue=True")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_model_f1_comparison.png", dpi=160)
    plt.close(fig)

    champion = models["random_forest_champion"].best_estimator_
    y_pred = champion.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No revenue", "Revenue"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Champion confusion matrix ({scenario})")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{scenario}_champion_confusion_matrix.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(champion, x_test, y_test, ax=ax)
    ax.set_title(f"Champion ROC curve ({scenario})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_champion_roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_estimator(champion, x_test, y_test, ax=ax)
    ax.set_title(f"Champion precision-recall curve ({scenario})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_champion_precision_recall_curve.png", dpi=160)
    plt.close(fig)

    save_roc_pr_comparison(scenario, models, x_test, y_test)


def save_feature_importance(scenario: str, champion: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    importance = permutation_importance(
        champion,
        x_test,
        y_test,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importances = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importances.insert(0, "scenario", scenario)
    importances.to_csv(OUTPUT_DIR / f"{scenario}_champion_permutation_importance.csv", index=False)

    top = importances.head(10).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#9a6f2d")
    ax.set_title(f"Champion permutation importance ({scenario})")
    ax.set_xlabel("Mean decrease in F1-score")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{scenario}_champion_feature_importance.png", dpi=160)
    plt.close(fig)
    return importances


def save_scenario_comparison(results: pd.DataFrame) -> None:
    champion_results = results[results["model"] == "random_forest_champion"].copy()
    champion_results.to_csv(OUTPUT_DIR / "champion_results_by_scenario.csv", index=False)

    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    comparison = champion_results.set_index("scenario")[metrics].T
    comparison["delta_with_minus_without"] = comparison["with_page_values"] - comparison["without_page_values"]
    comparison.to_csv(OUTPUT_DIR / "page_values_impact.csv")

    plot_data = champion_results.set_index("scenario")[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
    ax = plot_data.plot(kind="bar", figsize=(10, 5), color=["#3f7f93", "#9a6f2d", "#6f7f8f", "#b85c38", "#385a64"])
    ax.set_title("Champion performance with vs without PageValues")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "champion_page_values_impact.png", dpi=160)
    plt.close()


def run_scenario(
    scenario: str,
    x_train_full: pd.DataFrame,
    x_test_full: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    drop_columns = drop_columns or []
    x_train = x_train_full.drop(columns=drop_columns)
    x_test = x_test_full.drop(columns=drop_columns)
    preprocessor = make_preprocessor(x_train)
    searches = build_models(preprocessor)

    rows = []
    reports = {}
    for name, search in searches.items():
        search.fit(x_train, y_train)
        rows.append(evaluate_model(scenario, name, search, x_test, y_test))
        reports[name] = classification_report(
            y_test,
            search.best_estimator_.predict(x_test),
            target_names=["No revenue", "Revenue"],
            zero_division=0,
        )

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(OUTPUT_DIR / f"{scenario}_model_results.csv", index=False)
    save_figures(scenario, results, searches, x_test, y_test)
    save_calibration_curves(scenario, searches, x_test, y_test)
    save_cv_scores_summary(scenario, searches)
    champion = searches["random_forest_champion"].best_estimator_
    importances = save_feature_importance(scenario, champion, x_test, y_test)
    save_learning_curves(scenario, searches, x_train, y_train)
    save_shap_summary(scenario, champion, x_test)
    threshold_summary = optimize_threshold(scenario, champion, x_test, y_test)
    return results, reports, importances, threshold_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_THRESHOLDS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["Revenue"] = df["Revenue"].astype(bool)
    save_dataset_summary(df)

    x = df.drop(columns=["Revenue"])
    y = df["Revenue"]
    x_train_full, x_test_full, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scenario_specs = {
        "with_page_values": [],
        "without_page_values": ["PageValues"],
    }

    all_results = []
    all_importances = []
    all_threshold_summaries = []
    all_reports = {}
    for scenario, drop_columns in scenario_specs.items():
        results, reports, importances, threshold_summary = run_scenario(
            scenario,
            x_train_full,
            x_test_full,
            y_train,
            y_test,
            drop_columns=drop_columns,
        )
        all_results.append(results)
        all_importances.append(importances)
        all_threshold_summaries.append(threshold_summary)
        all_reports[scenario] = reports

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results = combined_results.sort_values(["scenario", "f1"], ascending=[True, False])
    combined_results.to_csv(OUTPUT_DIR / "model_results_by_scenario.csv", index=False)
    pd.concat(all_importances, ignore_index=True).to_csv(OUTPUT_DIR / "champion_permutation_importance_by_scenario.csv", index=False)
    threshold_summaries = pd.concat(all_threshold_summaries, ignore_index=True)
    threshold_summaries.to_csv(OUTPUT_DIR / "threshold_summary_by_scenario.csv", index=False)
    save_business_value_plot(threshold_summaries)
    save_scenario_comparison(combined_results)

    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as file:
        for scenario, reports in all_reports.items():
            file.write(f"\n{scenario}\n")
            file.write("=" * len(scenario) + "\n")
            for name, report in reports.items():
                file.write(f"\n{name}\n")
                file.write("-" * len(name) + "\n")
                file.write(report)
                file.write("\n")

    # Backward-compatible filenames for the primary champion experiment.
    with_page_values = combined_results[combined_results["scenario"] == "with_page_values"].copy()
    with_page_values.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    with_page_importance = pd.concat(all_importances, ignore_index=True)
    with_page_importance = with_page_importance[with_page_importance["scenario"] == "with_page_values"].copy()
    with_page_importance.to_csv(OUTPUT_DIR / "champion_permutation_importance.csv", index=False)

    MAIN_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _main_copies = {
        "with_page_values_model_f1_comparison.png": "05_model_f1_comparison.png",
        "with_page_values_champion_confusion_matrix.png": "06_champion_confusion_matrix.png",
        "with_page_values_threshold_tuning.png": "07_threshold_tuning_with_page_values.png",
        "business_value_by_threshold.png": "08_business_value_by_threshold.png",
        "champion_page_values_impact.png": "04_champion_page_values_impact.png",
        "with_page_values_roc_pr_comparison.png": "09_roc_pr_comparison_all_models.png",
        "with_page_values_calibration_curves.png": "10_calibration_curves.png",
        "with_page_values_cv_scores_summary.png": "11_cv_scores_summary.png",
    }
    for src_name, dst_name in _main_copies.items():
        src = FIGURE_DIR / src_name
        if src.exists():
            shutil.copy2(src, MAIN_FIGURE_DIR / dst_name)
    _appendix_model_copies = {
        "with_page_values_champion_feature_importance.png": "A6_champion_feature_importance.png",
        "with_page_values_champion_precision_recall_curve.png": "A7_champion_precision_recall_curve.png",
        "with_page_values_champion_roc_curve.png": "A8_champion_roc_curve.png",
        "without_page_values_threshold_tuning.png": "A9_threshold_tuning_without_page_values.png",
        "without_page_values_champion_feature_importance.png": "A10_feature_importance_without_page_values.png",
        "without_page_values_champion_confusion_matrix.png": "A11_confusion_matrix_without_page_values.png",
        "without_page_values_champion_precision_recall_curve.png": "A12_precision_recall_without_page_values.png",
        "without_page_values_champion_roc_curve.png": "A13_roc_without_page_values.png",
        "without_page_values_model_f1_comparison.png": "A14_model_f1_without_page_values.png",
        "with_page_values_learning_curves.png": "A15_learning_curves_with_page_values.png",
        "without_page_values_learning_curves.png": "A16_learning_curves_without_page_values.png",
        "with_page_values_champion_shap.png": "A17_champion_shap.png",
        "with_page_values_cv_scores_summary.png": "A18_cv_scores_summary.png",
        "without_page_values_calibration_curves.png": "A19_calibration_curves_without_page_values.png",
    }
    for src_name, dst_name in _appendix_model_copies.items():
        src = FIGURE_DIR / src_name
        if src.exists():
            shutil.copy2(src, APPENDIX_MODEL_FIGURE_DIR / dst_name)

    # ── Copy canonical CSV outputs ────────────────────────────────────────────
    _main_csv_copies = {
        "target_distribution.csv": "02_target_distribution.csv",
        "model_results_by_scenario.csv": "03_model_results_by_scenario.csv",
        "champion_results_by_scenario.csv": "04_champion_results_by_scenario.csv",
        "page_values_impact.csv": "05_page_values_impact.csv",
        "threshold_summary_by_scenario.csv": "06_threshold_summary_by_scenario.csv",
        "business_cost_matrix.csv": "07_business_cost_matrix.csv",
        "business_threshold_summary.csv": "08_business_threshold_summary.csv",
    }
    for src_name, dst_name in _main_csv_copies.items():
        src = OUTPUT_DIR / src_name
        if src.exists():
            shutil.copy2(src, MAIN_OUTPUT_DIR / dst_name)

    _appendix_model_csv_copies = {
        "with_page_values_model_results.csv": "A11_model_results_with_page_values.csv",
        "without_page_values_model_results.csv": "A13_without_page_values_model_results.csv",
        "with_page_values_champion_permutation_importance.csv": "A14_champion_permutation_importance.csv",
        "champion_permutation_importance_by_scenario.csv": "A15_champion_permutation_importance_by_scenario.csv",
        "with_page_values_champion_permutation_importance.csv": "A16_with_page_values_feature_importance.csv",
        "without_page_values_champion_permutation_importance.csv": "A17_without_page_values_feature_importance.csv",
        "classification_reports.txt": "A18_classification_reports.txt",
        "with_page_values_cv_scores_summary.csv": "A23_cv_scores_summary.csv",
        "without_page_values_cv_scores_summary.csv": "A24_cv_scores_summary_without_page_values.csv",
    }
    for src_name, dst_name in _appendix_model_csv_copies.items():
        src = OUTPUT_DIR / src_name
        if src.exists():
            shutil.copy2(src, APPENDIX_MODEL_OUTPUT_DIR / dst_name)

    _threshold_csv_copies = {
        "with_page_values_threshold_results.csv": "A19_with_page_values_threshold_results.csv",
        "without_page_values_threshold_results.csv": "A20_without_page_values_threshold_results.csv",
        "with_page_values_threshold_summary.csv": "A21_with_page_values_threshold_summary.csv",
        "without_page_values_threshold_summary.csv": "A22_without_page_values_threshold_summary.csv",
    }
    for src_name, dst_name in _threshold_csv_copies.items():
        src = OUTPUT_DIR / src_name
        if src.exists():
            shutil.copy2(src, APPENDIX_THRESHOLDS_DIR / dst_name)

    print(combined_results.to_string(index=False))


if __name__ == "__main__":
    main()
