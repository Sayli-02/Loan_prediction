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

## Project structure

```text
loan-eligibility-prediction/
├── app.py                       # Streamlit web application
├── data.ipynb                   # Exploratory data analysis and model training
├── loan_approval_dataset.csv    # Historical loan application dataset
├── loan_model.pkl               # Trained Random Forest classifier
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files to keep out of Git
└── README.md                    # Project documentation
```

## Tech stack

- **Python** — application and model workflow
- **Streamlit** — interactive web dashboard
- **Pandas / NumPy** — data loading and preparation
- **Matplotlib / Seaborn** — charts and exploratory analysis
- **scikit-learn** — Logistic Regression, Decision Tree, Random Forest, and evaluation
- **Joblib** — loading the saved model
- **PyPDF2** — extracting text from structured PDF applications

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

## Author

**Yash Joshi**  
GitHub: [@YASHU-JOSHI](https://github.com/YASHU-JOSHI)

---

If you find this project useful, feel free to ⭐ the repository.
