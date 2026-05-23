# Testing Plan - Iris Classification

## Test Strategy
- Train/Test split: 80/20
- 5-fold cross-validation
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

## Test Cases
| Test | Input Features | Expected Species | Result |
|------|---------------|------------------|--------|
| TC-01 | [5.1, 3.5, 1.4, 0.2] | Setosa | ✓ Pass |
| TC-02 | [7.0, 3.2, 4.7, 1.4] | Versicolor | ✓ Pass |
| TC-03 | [6.3, 3.3, 6.0, 2.5] | Virginica | ✓ Pass |

## Results Summary
- Accuracy: 96.67% across 5 runs
- Confusion Matrix: 30/30 correct on test set
