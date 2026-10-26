# Loan Eligibility Prediction System

A machine-learning dashboard built with **Python, Streamlit, Pandas, Matplotlib, and scikit-learn** to explore historical loan data and predict whether an applicant's loan is likely to be approved or rejected.

The project includes an interactive web app, an exploratory/model-training notebook, a loan approval dataset, and a saved Random Forest model.

## Features

- **Overview dashboard** — summarizes application counts, approvals, rejections, and historical approval rate.
- **Dataset preview** — inspect records, column types, missing values, duplicates, and descriptive statistics.
- **Manual prediction** — enter one applicant's details and view the predicted class and model probability estimates.
- **Bulk prediction** — upload applicant records as CSV, Excel, JSON, or a structured text-based PDF; review predictions and download results as CSV.
- **Analytics** — visualize loan status, CIBIL score, education, employment, income, and loan amount patterns.
- **Input validation** — reports missing required columns and missing or invalid values before prediction.

## What I added (College ML Mini Project Extensions)

This repository extends the original project with **3 core machine learning features** plus an optional risk segmentation module:

### 1. 5-Model Benchmark & Comparison
- Expanded the comparison from 3 to **5 classifiers**: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (`SVC(probability=True)`), and Gaussian Naive Bayes.
- Distance-sensitive models (Logistic Regression and SVM) use `StandardScaler` inside a scikit-learn `Pipeline` to prevent data leakage, while tree-based models and Naive Bayes train on unscaled data.
- Computed **5 evaluation metrics** across all models: **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
- Exported the results to `results/model_comparison.csv` and full ROC curve coordinates to `results/roc.json`.
- Built a dedicated **"Model Comparison"** page in the Streamlit application featuring the performance table, an interactive Matplotlib ROC curve plot, and key takeaways.

### 2. Random Forest Hyperparameter Tuning (`GridSearchCV`)
- Performed a 3-fold cross-validation grid search (`n_estimators: [100, 200]`, `max_depth: [5, 10, None]`, `min_samples_split: [2, 5]`) targeting F1 optimization.
- Addressed dataset class imbalance (Approved 62.2% vs Rejected 37.8%) by configuring `class_weight="balanced"`.
- Displayed before-and-after performance metrics and best parameters on the Model Comparison dashboard.
- Archived the original model to `loan_model_original.pkl` and implemented the strict persistence rule (retaining the original model since default test F1 = 0.9869 &ge; tuned test F1 = 0.9859).

### 3. Explainable AI with SHAP (Local & Global Explanations)
- Integrated `shap.TreeExplainer` on the **Manual Prediction** page to provide transparent, interpretable reasons behind each individual loan decision.
- Handled SHAP library version differences across 2D/3D output shapes (`n_samples, n_features, n_classes`) to reliably extract class 1 (Approved) attribution vectors.
- Displayed sorted horizontal contribution charts with intuitive green (pushing toward approval) and red (pushing toward rejection) color-coding and plain-English explanations.
- Cached the explainer with `st.cache_resource` for real-time responsiveness.
- Added a global **Feature Importance** chart on the Model Comparison page reflecting Gini impurity reduction across all trees.

### 4. Optional Extension: K-Means Risk Segmentation
- Fitted an unsupervised 3-cluster `KMeans` model on standardized numeric features: annual income, loan amount, CIBIL score, and total asset value.
- Labeled segments into **Low Risk** (99.3% historical approval rate, mean CIBIL 757), **Medium Risk** (63.0% approval rate, mean CIBIL 597), and **High Risk** (24.0% approval rate, mean CIBIL 447).
- Integrated real-time risk segment tags into Manual Prediction and added cluster distribution and approval rate charts to the Analytics page.

## Project structure

```text
loan-eligibility-prediction/
├── app.py                       # Extended Streamlit web application
├── data.ipynb                   # EDA, 5-model comparison & GridSearchCV notebook
├── loan_approval_dataset.csv    # Historical loan application dataset (4,269 records)
├── loan_model.pkl               # Production Random Forest classifier
├── loan_model_original.pkl      # Backup of original trained Random Forest
├── requirements.txt             # Python dependencies (including shap)
├── .gitignore                   # Files to keep out of Git
├── README.md                    # Project documentation
└── results/                     # Generated benchmark artifacts
    ├── model_comparison.csv     # 5-model benchmark metrics table
    ├── roc.json                 # ROC curve FPR/TPR points & AUC values
    ├── tuning_info.json         # GridSearchCV parameters & before/after deltas
    ├── kmeans_model.pkl         # Fitted 3-cluster KMeans model
    ├── kmeans_scaler.pkl        # Fitted StandardScaler for clustering
    └── kmeans_risk_map.pkl      # Cluster index to risk label mapping
```

## Tech stack

- **Python 3.11** — application and model workflow
- **Streamlit** — interactive web dashboard
- **Pandas / NumPy** — data manipulation and feature engineering
- **scikit-learn** — Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, Pipelines, GridSearchCV, and KMeans
- **SHAP** — TreeExplainer for interpretable local feature attribution
- **Matplotlib / Seaborn** — ROC curves, SHAP charts, probability and analytics plots
- **Joblib** — model serialization and deserialization
- **PyPDF2** — extracting applicant text from structured PDF files

## Dataset

The supplied CSV contains **4,269 records** and 13 columns, including the loan ID, applicant information, financial/asset values, and historical loan status. The app uses 11 model features; `loan_id` and the target `loan_status` are not model inputs.

The 11 expected feature columns are:

| Column | Description |
|---|---|
| `no_of_dependents` | Number of dependents |
| `education` | `Graduate` or `Not Graduate` |
| `self_employed` | `Yes` or `No` |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan repayment term |
| `cibil_score` | Credit score |
| `residential_assets_value` | Residential asset value |
| `commercial_assets_value` | Commercial asset value |
| `luxury_assets_value` | Luxury asset value |
| `bank_asset_value` | Bank asset value |

The training notebook maps education and self-employment categories to numeric values and encodes the target as `Approved = 1` and `Rejected = 0`. It compares Logistic Regression, Decision Tree, and Random Forest models, and saves the Random Forest classifier to `loan_model.pkl`.

## Getting started

### 1. Requirements

Use Python 3.10 or newer if possible. Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

Keep `app.py`, `loan_model.pkl`, and `loan_approval_dataset.csv` in the same directory, then run:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Bulk upload formats

Bulk Prediction accepts the following formats:

- **CSV (`.csv`)** — first row must contain column names.
- **Excel (`.xlsx`, `.xls`)** — first worksheet is read; `openpyxl` is needed for `.xlsx`.
- **JSON (`.json`)** — a list of applicant objects, a single applicant object, or a list nested under `data`, `records`, or `applicants`.
- **PDF (`.pdf`)** — selectable text with values written as `column_name: value`; intended for one applicant per PDF. Scanned/image-only PDFs may not work.

Every bulk input must include all 11 model feature columns. For categorical fields, use exactly `Graduate` / `Not Graduate` and `Yes` / `No`. The app provides a downloadable `loan_prediction_results.csv` containing the input records, predicted class, and approval probability.

## Re-training the model

Open `data.ipynb` in Jupyter Notebook or JupyterLab and run the cells in order. The notebook performs exploratory analysis, prepares the features and target, compares three classifiers, evaluates the Random Forest, and saves `loan_model.pkl`.

If you replace the model, make sure the replacement expects the same 11 features in the same order and uses class `1` for approval and class `0` for rejection, or update the app accordingly.

## Important limitations

- This project is an educational decision-support demonstration, not a lending decision system validated for real-world use.
- Model probabilities are estimates from the trained classifier and are not a guarantee that a loan will be approved.
- Historical data may contain bias or reflect patterns that do not generalize to new applicants.
- Do not use this model as the sole basis for approving or rejecting a real loan. A qualified reviewer and the lender's policies should be involved.
- Only load `loan_model.pkl` from a trusted source. Pickle/joblib files can execute code when deserialized.

## Credits and Author

- **Original Author:** Yash Joshi ([github.com/YASHU-JOSHI/Loan_prediction](https://github.com/YASHU-JOSHI/Loan_prediction))
- **College ML Mini Project Extension:** Added 5-model benchmark, GridSearchCV tuning, SHAP explainability, and K-Means risk segments.

---

If you find this project useful, feel free to ⭐ the repository.
