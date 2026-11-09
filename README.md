# 🛡️ Fraudulent Transaction Detection - Ensemble ML Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Ensemble-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

An interactive Streamlit application that explores transaction data, compares ensemble classification algorithms, and predicts whether a transaction is potentially fraudulent.

## 🚀 Features

- Dataset overview, preview, and descriptive statistics
- Fraud and legitimate transaction distribution
- Feature correlation heatmap
- Transaction-pattern visualizations
- Five classification models
- Accuracy, balanced accuracy, precision, recall, and F1-score comparison
- Confusion matrix and classification report
- Interactive transaction prediction form
- Adjustable training sample size and test split

## 🤖 Machine Learning Models

| Model | Purpose |
|---|---|
| Decision Tree | Rule-based classification |
| Bagging Classifier | Combines estimators trained on bootstrap samples |
| AdaBoost Classifier | Sequential boosting |
| Random Forest | Combines multiple decision trees |
| Voting Classifier | Combines the four base classifier types using hard voting |

The Voting Classifier uses majority voting. Class weights are used in the Decision Tree and Random Forest components to help address class imbalance.

## 📊 Dataset

The application expects the file `card_transdata.csv` in the same folder as `app.py`.

The dataset should contain these columns:

| Feature | Description |
|---|---|
| `distance_from_home` | Distance from the cardholder's home |
| `distance_from_last_transaction` | Distance from the previous transaction |
| `ratio_to_median_purchase_price` | Purchase-price ratio relative to the median |
| `repeat_retailer` | Whether the retailer was visited before |
| `used_chip` | Whether a chip was used |
| `used_pin_number` | Whether a PIN was used |
| `online_order` | Whether the transaction was an online order |
| `fraud` | Target: 0 = legitimate, 1 = fraud |

Column names are converted to lowercase and trimmed before validation. Rows with missing or non-finite values are removed.

### Large dataset handling

To keep the app responsive, the sidebar allows a stratified sample of up to 300,000 records to be used for training and evaluation. The source CSV remains unchanged. Increase the sample size when your computer or deployment resources can handle longer training times.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive dashboard |
| Pandas | CSV loading and data processing |
| NumPy | Numerical operations |
| Matplotlib | Charts |
| Seaborn | Statistical visualizations |
| Scikit-learn | Classification models and evaluation |

## 🖥️ Dashboard Sections

1. **Dataset Overview** – Preview transactions, inspect descriptive statistics, and view class distribution.
2. **Fraud Insights** – Explore correlations, purchase-ratio distributions, and online-order fraud rates.
3. **Model Comparison** – Compare models with accuracy, balanced accuracy, precision, recall, F1-score, confusion matrices, and classification reports.
4. **Transaction Predictor** – Enter transaction attributes and select a trained model to classify the transaction.

## 📁 Project Structure

```text
Fraudulent-Transaction-Detection/
│
├── app.py
├── card_transdata.csv
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
└── screenshots/
```

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 or newer
- pip
- `card_transdata.csv`

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Fraudulent-Transaction-Detection
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Git Bash:**

```bash
python -m venv venv
source venv/Scripts/activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place `card_transdata.csv` beside `app.py`.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## ⚠️ Limitations

- Model results depend on the supplied dataset and selected stratified sample.
- Fraud detection is an imbalanced classification problem; accuracy alone is not sufficient.
- The sample size setting trades training time against the amount of data used.
- The prediction form expects the seven feature values in the dataset's numeric encoding.
- No live banking connection, transaction blocking, or real-time alerting is implemented.
- This is an educational demonstration, not a production financial fraud detection system.

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**  
GitHub: [@bemo-codes](https://github.com/bemo-codes)

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<p align="center">Built with Python 🐍, Streamlit ⚡ and Scikit-learn 🤖</p>
