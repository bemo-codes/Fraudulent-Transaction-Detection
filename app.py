###############################################################################
#
#   Fraudulent Transaction Detection - Ensemble Machine Learning Dashboard
#
###############################################################################

# =============================================================================
#   Importing required libraries
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
#   Application configuration
# =============================================================================

st.set_page_config(
    page_title="Fraudulent Transaction Detection",
    page_icon="🛡️",
    layout="wide",
)


# =============================================================================
#   Constants
# =============================================================================

DATA_FILE = "card_transdata.csv"
TARGET_COLUMN = "fraud"

FEATURE_COLUMNS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]

BINARY_FEATURES = [
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]


# =============================================================================
#   Function name: load_data
#   Input: data_path (string)
#   Description: Loads the transaction dataset and caches it for reuse.
#   Output: Cleaned DataFrame
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(data_path):
    df = pd.read_csv(data_path, low_memory=False)

    # Standardize column names without changing feature meaning.
    df.columns = df.columns.str.strip().str.lower()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Keep only the columns used by this application.
    df = df[required_columns].copy()

    # Convert all expected columns to numeric values.
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # The target must contain only the two expected classes.
    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


# =============================================================================
#   Function name: make_sample
#   Input: df (DataFrame), sample_size (int), random_state (int)
#   Description: Creates a reproducible stratified sample for faster training.
#   Output: Sampled DataFrame
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def make_sample(df, sample_size, random_state=42):
    if sample_size >= len(df):
        return df.copy()

    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    return sampled_df


# =============================================================================
#   Function name: build_models
#   Input: None
#   Description: Creates the five classification models used in the dashboard.
#   Output: Dictionary containing untrained model objects
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def build_models():
    decision_tree = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        n_estimators=10,
        max_samples=0.8,
        random_state=42,
        n_jobs=-1,
    )

    adaboost = AdaBoostClassifier(
        n_estimators=40,
        learning_rate=0.5,
        random_state=42,
    )

    random_forest = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    voting = VotingClassifier(
        estimators=[
            (
                "decision_tree",
                DecisionTreeClassifier(
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
            (
                "bagging",
                BaggingClassifier(
                    estimator=DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                    n_estimators=10,
                    max_samples=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "adaboost",
                AdaBoostClassifier(
                    n_estimators=40,
                    learning_rate=0.5,
                    random_state=42,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=18,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        voting="hard",
        n_jobs=-1,
    )

    return {
        "Decision Tree": decision_tree,
        "Bagging Classifier": bagging,
        "AdaBoost Classifier": adaboost,
        "Random Forest": random_forest,
        "Voting Classifier": voting,
    }


# =============================================================================
#   Function name: train_models
#   Input: X_train (DataFrame), y_train (Series)
#   Description: Trains and caches all classification models.
#   Output: Dictionary containing trained model objects
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train):
    models = build_models()

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# =============================================================================
#   Function name: evaluate_models
#   Input: models (dict), X_test (DataFrame), y_test (Series)
#   Description: Calculates classification metrics for every trained model.
#   Output: Results DataFrame and prediction dictionary
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def evaluate_models(models, X_test, y_test):
    results = []
    predictions = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred

        results.append(
            {
                "Model": model_name,
                "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
                "Balanced Accuracy (%)": (
                    balanced_accuracy_score(y_test, y_pred) * 100
                ),
                "Precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, y_pred, zero_division=0
                ),
            }
        )

    return pd.DataFrame(results), predictions


# =============================================================================
#   Function name: get_feature_defaults
#   Input: df (DataFrame)
#   Description: Creates sensible defaults and ranges for prediction inputs.
#   Output: Dictionary of feature ranges and default values
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def get_feature_defaults(df):
    settings = {}

    for column in FEATURE_COLUMNS:
        low = float(df[column].quantile(0.01))
        high = float(df[column].quantile(0.99))
        median = float(df[column].median())

        if low == high:
            high = low + 1.0

        settings[column] = {
            "min": max(0.0, low),
            "max": max(high, low + 0.1),
            "value": min(max(median, max(0.0, low)), max(high, low + 0.1)),
        }

    return settings


# =============================================================================
#   Function name: main
#   Input: None
#   Description: Builds the Streamlit dashboard and manages the fraud workflow.
#   Output: Streamlit application
#   Author: Shivam Ramesh Kurlekar
# =============================================================================

def main():

    st.title("🛡️ Fraudulent Transaction Detection")
    st.caption(
        "Ensemble Machine Learning Dashboard | Transaction Classification"
    )

    st.write(
        "Explore transaction patterns, compare ensemble classifiers, "
        "and test transaction records using machine learning."
    )

    # -------------------------------------------------------------------------
    #   Load dataset
    # -------------------------------------------------------------------------

    data_path = Path(__file__).resolve().parent / DATA_FILE

    if not data_path.exists():
        st.error(
            f"`{DATA_FILE}` was not found. Place the CSV file in the same "
            "folder as `app.py`."
        )
        st.stop()

    try:
        df = load_data(str(data_path))
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()

    if df.empty:
        st.error("The dataset has no usable records after cleaning.")
        st.stop()

    class_counts = df[TARGET_COLUMN].value_counts()
    if len(class_counts) != 2 or not {0, 1}.issubset(set(class_counts.index)):
        st.error("The `fraud` column must contain both labels: 0 and 1.")
        st.stop()

    # -------------------------------------------------------------------------
    #   Sidebar configuration
    # -------------------------------------------------------------------------

    st.sidebar.header("⚙️ Model Configuration")

    maximum_sample = min(len(df), 300_000)
    if len(df) > 50_000:
        sample_size = st.sidebar.select_slider(
            "Records used for model training",
            options=[
                value
                for value in [50_000, 100_000, 200_000, 300_000]
                if value <= maximum_sample
            ],
            value=min(100_000, maximum_sample),
            help=(
                "A stratified sample is used to keep training responsive. "
                "The source CSV is not modified."
            ),
        )
    else:
        sample_size = len(df)
        st.sidebar.info(
            f"Using all {len(df):,} available records."
        )

    test_percentage = st.sidebar.slider(
        "Test set size (%)",
        min_value=10,
        max_value=30,
        value=20,
        step=5,
    )

    # -------------------------------------------------------------------------
    #   Sample, split and train models
    # -------------------------------------------------------------------------

    with st.spinner(
        "Preparing transaction sample and training ensemble models..."
    ):
        sampled_df = make_sample(df, int(sample_size))

        X = sampled_df[FEATURE_COLUMNS]
        y = sampled_df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percentage / 100,
            random_state=42,
            stratify=y,
        )

        models = train_models(X_train, y_train)
        results, predictions = evaluate_models(
            models, X_test, y_test
        )

    # -------------------------------------------------------------------------
    #   Main metrics
    # -------------------------------------------------------------------------

    fraud_count = int((df[TARGET_COLUMN] == 1).sum())
    legitimate_count = int((df[TARGET_COLUMN] == 0).sum())
    fraud_rate = fraud_count / len(df) * 100

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total Transactions", f"{len(df):,}")
    metric2.metric("Fraudulent", f"{fraud_count:,}")
    metric3.metric("Legitimate", f"{legitimate_count:,}")
    metric4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    if len(sampled_df) < len(df):
        st.info(
            f"Training uses a stratified sample of {len(sampled_df):,} "
            f"records from {len(df):,} available records. "
            f"The held-out test set contains {len(X_test):,} records."
        )

    # -------------------------------------------------------------------------
    #   Dashboard navigation
    # -------------------------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📁 Dataset Overview",
            "📊 Fraud Insights",
            "🤖 Model Comparison",
            "🔎 Transaction Predictor",
        ]
    )

    # =========================================================================
    #   Tab 1: Dataset Overview
    # =========================================================================

    with tab1:
        st.subheader("Transaction Dataset")

        left, right = st.columns([2, 1])

        with left:
            st.write("**Dataset Preview**")
            st.dataframe(df.head(15), use_container_width=True)

        with right:
            st.write("**Fraud Distribution**")
            distribution = (
                df[TARGET_COLUMN]
                .value_counts()
                .reindex([0, 1], fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(
                ["Legitimate", "Fraud"],
                distribution.values,
            )
            ax.set_ylabel("Number of transactions")
            ax.set_title("Target Distribution")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.write("**Descriptive Statistics**")
        st.dataframe(df.describe().T, use_container_width=True)

        st.write("**Missing Values After Cleaning**")
        missing_values = df.isna().sum().rename("Missing Values")
        st.dataframe(missing_values.to_frame(), use_container_width=True)

    # =========================================================================
    #   Tab 2: Fraud Insights
    # =========================================================================

    with tab2:
        st.subheader("Feature Correlation")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Transaction Amount Ratio by Fraud Label")

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(
            data=df,
            x="ratio_to_median_purchase_price",
            hue="fraud",
            bins=40,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_xlabel("Ratio to Median Purchase Price")
        ax.set_title("Purchase Ratio Distribution")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Fraud Rate by Transaction Type")

        fraud_by_feature = (
            df.groupby("online_order")[TARGET_COLUMN]
            .mean()
            .mul(100)
            .rename(index={0: "Offline", 1: "Online"})
        )
        st.bar_chart(fraud_by_feature)

    # =========================================================================
    #   Tab 3: Model Comparison
    # =========================================================================

    with tab3:
        st.subheader("Classification Model Performance")

        st.dataframe(
            results.style.format(
                {
                    "Accuracy (%)": "{:.2f}",
                    "Balanced Accuracy (%)": "{:.2f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1 Score": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "For imbalanced fraud data, compare recall, precision, F1-score "
            "and balanced accuracy instead of relying on accuracy alone."
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        results_sorted = results.sort_values("F1 Score", ascending=False)
        sns.barplot(
            data=results_sorted,
            x="Model",
            y="F1 Score",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_title("F1 Score Comparison")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        selected_model = st.selectbox(
            "Select a model to inspect",
            list(models.keys()),
        )

        selected_predictions = predictions[selected_model]
        matrix = confusion_matrix(
            y_test,
            selected_predictions,
            labels=[0, 1],
        )

        left, right = st.columns(2)

        with left:
            st.write("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.write("**Classification Report**")
            report = classification_report(
                y_test,
                selected_predictions,
                labels=[0, 1],
                target_names=["Legitimate", "Fraud"],
                zero_division=0,
            )
            st.code(report)

    # =========================================================================
    #   Tab 4: Transaction Predictor
    # =========================================================================

    with tab4:
        st.subheader("Analyze a Transaction")
        st.write(
            "Enter the transaction characteristics and select a trained "
            "model to classify the record."
        )

        model_choice = st.selectbox(
            "Select prediction model",
            list(models.keys()),
            key="prediction_model",
        )

        feature_settings = get_feature_defaults(df)

        with st.form("fraud_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                distance_home = st.number_input(
                    "Distance from Home",
                    min_value=feature_settings["distance_from_home"]["min"],
                    max_value=feature_settings["distance_from_home"]["max"],
                    value=feature_settings["distance_from_home"]["value"],
                )

                distance_last = st.number_input(
                    "Distance from Last Transaction",
                    min_value=feature_settings[
                        "distance_from_last_transaction"
                    ]["min"],
                    max_value=feature_settings[
                        "distance_from_last_transaction"
                    ]["max"],
                    value=feature_settings[
                        "distance_from_last_transaction"
                    ]["value"],
                )

                purchase_ratio = st.number_input(
                    "Ratio to Median Purchase Price",
                    min_value=feature_settings[
                        "ratio_to_median_purchase_price"
                    ]["min"],
                    max_value=feature_settings[
                        "ratio_to_median_purchase_price"
                    ]["max"],
                    value=feature_settings[
                        "ratio_to_median_purchase_price"
                    ]["value"],
                )

                repeat_retailer = st.selectbox(
                    "Repeat Retailer",
                    options=[1, 0],
                    format_func=lambda value: "Yes" if value == 1 else "No",
                )

            with col2:
                used_chip = st.selectbox(
                    "Used Chip",
                    options=[1, 0],
                    format_func=lambda value: "Yes" if value == 1 else "No",
                )

                used_pin = st.selectbox(
                    "Used PIN Number",
                    options=[1, 0],
                    format_func=lambda value: "Yes" if value == 1 else "No",
                )

                online_order = st.selectbox(
                    "Online Order",
                    options=[1, 0],
                    format_func=lambda value: "Yes" if value == 1 else "No",
                )

            submitted = st.form_submit_button(
                "Analyze Transaction",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            transaction = pd.DataFrame(
                [
                    {
                        "distance_from_home": distance_home,
                        "distance_from_last_transaction": distance_last,
                        "ratio_to_median_purchase_price": purchase_ratio,
                        "repeat_retailer": repeat_retailer,
                        "used_chip": used_chip,
                        "used_pin_number": used_pin,
                        "online_order": online_order,
                    }
                ],
                columns=FEATURE_COLUMNS,
            )

            prediction = int(
                models[model_choice].predict(transaction)[0]
            )

            st.divider()

            if prediction == 1:
                st.error(
                    "🚨 Model Result: Potentially Fraudulent Transaction"
                )
                st.warning(
                    "This is an automated model classification. "
                    "It is not proof that a transaction is fraudulent."
                )
            else:
                st.success(
                    "✅ Model Result: Classified as Legitimate"
                )
                st.info(
                    "The selected model classified this transaction "
                    "as non-fraudulent."
                )

            st.caption(f"Model used: {model_choice}")

    # -------------------------------------------------------------------------
    #   Application footer
    # -------------------------------------------------------------------------

    st.divider()
    st.caption(
        "Educational machine learning project. Predictions should not be "
        "used as the sole basis for financial or fraud-investigation decisions."
    )


# =============================================================================
#   Application entry point
# =============================================================================

if __name__ == "__main__":
    main()
