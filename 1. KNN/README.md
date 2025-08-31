# k-Nearest Neighbors (KNN)

KNN is a **lazy**, **non-parametric** classifier.  
For a query point **x**, compute distances to all training points, take the *k* closest, and return the majority label.

---

## Distance Metric

**Euclidean Distance** is commonly used:

\[
d(x, p) = \sqrt{\sum_i (x_i - p_i)^2}
\]

Other distance metrics are possible.

---

## Hyperparameters

- **k**: Number of neighbors (odd values reduce ties)

---

## Complexity

- **Prediction:** \( O(n \cdot d) \) per sample  
  *(n = train size, d = number of features)*

---

## Preprocessing

- **Scale features** (e.g., `StandardScaler`) so distances are meaningful

---

## Evaluation

- Use **accuracy**, **confusion matrix**, **ROC** (for binary classification)
- Use **cross-validation** to select *k*

---

## Common Pitfalls

- Returning the wrong shape (must be `(n_samples,)` for labels)
- Not scaling features → poor results when feature scales differ
- Ties: for even *k*, ties can occur (resolve by smaller average distance or pick smallest label)