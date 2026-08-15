# Financial Distress Stress Test

## Question
Can a bankruptcy prediction model remain reliable when selected financial
indicators deteriorate?

## Dataset
Taiwanese Company Bankruptcy Prediction (Kaggle), 6,819 firms, 96 features,
0 missing values, 0 duplicates. Target `Bankrupt?` is heavily imbalanced
(3.2% bankrupt).

## Objective
Test the sensitivity of a trained model's performance to a simulated
financial deterioration scenario, without retraining the model.

## Method
1. Stratified 80/20 train-test split (random_state=42)
2. Baseline: RandomForestClassifier(class_weight='balanced'), trained once
3. Four financial indicators selected and stressed at 0%, 10%, 20%, 30%
   relative change (documented direction and rationale in the notebook)
4. Same trained model evaluated at each stress level — never retrained
5. Metrics compared: accuracy, precision, recall, F1, false negatives

## Stress variables
| Variable | Direction | Why |
|---|---|---|
| ROA(C) | decrease | lower profitability = weaker performance |
| Cash/Total Assets | decrease | lower liquidity = less shock resilience |
| Debt ratio % | increase | higher leverage = greater burden |
| Borrowing dependency | increase | more reliance on borrowing = more pressure |

## Results
| Stress % | Accuracy | Precision | Recall | F1 | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0% | 97.2% | 80.0% | 18.2% | 29.6% | 36 |
| 10% | 96.9% | 57.1% | 18.2% | 27.6% | 36 |
| 20% | 97.1% | 64.3% | 20.5% | 31.0% | 35 |
| 30% | 97.0% | 61.5% | 18.2% | 28.1% | 36 |

Recall stayed essentially flat across the stress ladder. Precision dropped
sharply — the model starts flagging more healthy companies as bankrupt as
conditions deteriorate.

## Limitations
- This is a simulated stress scenario, not evidence of real-world crisis behavior.
- The stress magnitude (10/20/30%) is a design choice, not an empirically derived crisis magnitude.
- Only 4 of 96 features were stressed; the model may rely more heavily on other features.
- Baseline recall (18.2%) is already low, which limits how much further deterioration this experiment could detect.
- Results may differ with another model, dataset, or stress design.

## What this does NOT prove
- It does not prove the model would fail in an actual economic crisis.
- It does not establish causality between the stressed variables and model behavior.

## Reproducing this
```
pip install -r requirements.txt
python notebook/01_financial_distress_stress_test.py
```
