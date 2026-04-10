import numpy as np

tp = 157060
fp = 38785 + 1232
fn = 822 + 55687
# tn = 235326 + 730 + 1416 + 113799

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
