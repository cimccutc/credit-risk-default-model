# Credit Risk Default Prediction
 
An end-to-end machine learning pipeline that predicts consumer loan default using a
Random Forest classifier built with scikit-learn. The project prioritizes a clean
data-quality workflow and model interpretability over raw predictive performance —
the goal is a model whose decisions a non-technical credit team could actually
understand and trust.
 
## Overview
 
Using a public dataset of 32,000+ consumer lending records, this project:
 
- Audits and resolves missing data with a documented imputation strategy
- One-hot encodes categorical loan attributes for modeling
- Trains an interpretable Random Forest classifier to predict loan default
- Extracts feature importances to explain what drives the model's risk assessment
## Dataset
 
[Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
— 32,581 records, 12 features. *(Confirm this is the same dataset you downloaded.)*
 
The CSV is **not** included in this repository, in line with Kaggle's terms. To run
the notebook, download `credit_risk_dataset.csv` from the link above and place it in
the project folder.
 
**Target variable:** `loan_status` (0 = loan repaid, 1 = default).
 
## Methodology
 
1. **Data quality audit** — Checked every column for missing values. Found 895 missing
   in `person_emp_length` and 3,116 in `loan_int_rate` (~4,000 total), and filled both
   using median imputation so the strategy is documented and reproducible.
2. **Encoding** — Converted the categorical fields (`person_home_ownership`,
   `loan_intent`, `cb_person_default_on_file`, `loan_grade`) into numeric features via
   one-hot encoding (`drop_first=True`).
3. **Train/test split** — 80/20 split, stratified on the target to preserve the ~22%
   default rate in both sets.
4. **Model** — `RandomForestClassifier` with `n_estimators=100` and `max_depth=5`. Tree
   depth is deliberately capped to keep the model interpretable and reduce overfitting,
   which matters more than squeezing out extra accuracy in a credit-risk context.
5. **Interpretation** — Extracted feature importances to translate the model's behavior
   into business-readable risk drivers.
## Results
 
On the held-out test set (6,517 records):
 
| Metric | Value |
|---|---|
| Overall accuracy | ~90% |
| Precision (default class) | ~96% |
| Recall (default class) | ~58% |
 
The model is highly reliable when it flags a loan as a default risk, but it misses
roughly 42% of actual defaulters — a direct consequence of the class imbalance and a
threshold tuned toward precision. Overall accuracy looks strong largely because
non-defaults dominate the data, so recall on the minority class is the more honest
measure of how much work is left (see Limitations).
 
### Top risk drivers (feature importance)
 
| Feature | Importance |
|---|---|
| Loan percent of income | 32.7% |
| Loan interest rate | 15.6% |
| Applicant income | 13.3% |
| Loan grade "D" | 12.2% |
 
These four features account for roughly three-quarters of the model's decisions. The
ranking mirrors real underwriting logic: the loan's size *relative to income* matters
more than raw income alone, and a $90k loan against a $100k salary is flagged as far
riskier than a $5k loan against a $50k salary.
 
## Limitations & next steps
 
- **Feature leakage.** Two of the top features — interest rate and loan grade — are
  assigned *by the lender* based on its own risk assessment of the borrower, so the
  model is partly learning "loans the lender already priced as risky are risky." A
  production version would separate application-time features from lender-assigned ones
  and re-evaluate on inputs available *before* a decision is made.
- **Class imbalance / recall.** Only ~58% of true defaults are caught. Worth testing
  `class_weight="balanced"`, decision-threshold tuning, and resampling, reported with
  precision-recall and ROC-AUC rather than accuracy.
- **Simple imputation.** Median imputation is a reasonable baseline; a next pass would
  investigate *why* employment length and interest rate are missing before deciding how
  to fill them.
- **No tuning / single split.** `max_depth` was chosen for interpretability, not
  optimized. Cross-validation and a hyperparameter search would make the results more
  robust.
- **Dataset scope.** This is a public, educational consumer-lending dataset and does not
  represent any specific lender's portfolio.
## Tech stack
 
Python, pandas, NumPy, scikit-learn, Jupyter
 
## Running it locally
 
```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook credit_risk_project.ipynb
```
 
Place `credit_risk_dataset.csv` in the same folder, then run the notebook top to bottom.
