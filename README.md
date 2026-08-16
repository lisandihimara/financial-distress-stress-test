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
5. Metrics compared: accuracy, precision, recall, F1, false negatives,
   predicted-class changes, and predicted bankruptcy probability (risk score)

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
| 0% | 97.3% | 81.8% | 20.5% | 32.7% | 35 |
| 10% | 97.0% | 60.0% | 20.5% | 30.5% | 35 |
| 20% | 97.0% | 60.0% | 20.5% | 30.5% | 35 |
| 30% | 97.1% | 64.3% | 20.5% | 31.0% | 35 |

At the classification level, recall and false negatives were completely
unchanged across every stress level — on its own this metric set would
suggest the model was unaffected by the stress scenario. Precision did
drop sharply (81.8% → 60.0–64.3%): under stress the model flags more
healthy companies as bankrupt.

## What the classification metrics alone would have missed
| Stress % | Classes changed (of 1,364) | Mean bankruptcy probability (full test set) | Mean probability among actually-bankrupt firms |
|---:|---:|---:|---:|
| 0% | — | 2.9% | 28.1% |
| 10% | 4 | 4.5% | 29.3% |
| 20% | 4 | 4.8% | 29.4% |
| 30% | 3 | 4.9% | 29.3% |

A small number of individual predictions did flip class under stress
(4 at 10–20%, 3 at 30%) - enough to explain why the class-level metrics
above look completely static, since these flips happened to cancel out
in the aggregate confusion matrix. More notably, the model's underlying
predicted bankruptcy *probability* rose with stress even where the final
class label didn't change: mean predicted risk across the full test set
rose ~66% relative (2.9% → 4.9%). Among firms that were actually
bankrupt, mean predicted risk rose from 28.1% at baseline to ~29.3% by
10% stress, then plateaued (29.4% at 20%, 29.3% at 30%) rather than
continuing to climb. A model that looks perfectly stable by its
classification output alone can still be quietly revising its risk
assessment upward.

## Limitations
- This is a simulated stress scenario, not evidence of real-world crisis behavior.
- The stress magnitude (10/20/30%) is a design choice, not an empirically derived crisis magnitude.
- Only 4 of 96 features were stressed; the model may rely more heavily on other features.
- Baseline recall (20.5%) is already low, which limits how much further classification-level deterioration this experiment could detect — this is part of why the probability-level check matters.
- The probability shifts observed are modest in absolute terms (a few percentage points).
- Exact metric values can shift slightly by scikit-learn version even with the same random_state and dataset. The results above were run in Google Colab; the notebook now prints the exact Python/scikit-learn/pandas/numpy versions used, so anyone reproducing this can check whether their environment matches.
- Results may differ with another model, dataset, or stress design.

## What this does NOT prove
- It does not prove the model would fail in an actual economic crisis.
- It does not establish causality between the stressed variables and model behavior.

## Reproducing this
```
pip install -r requirements.txt
python notebook/01_financial_distress_stress_test.py
```
