# MLB All-Star Predictor with TRNCG Optimization
A PyTorch-based machine learning project that predicts future MLB All-Star players using historical performance data. This project demonstrates the use of the **Trust Region Newton Conjugate Gradient (TRNCG)** optimization algorithm to efficiently train a neural network by minimizing the Binary Cross Entropy (BCE) loss.

---

## 🔍 Project Overview
We developed a binary classifier that predicts whether an MLB player will be selected as an All-Star in a given season. The project compares the performance of TRNCG against Stochastic Gradient Descent (SGD), showcasing faster convergence and fewer forward/backward propagations.

---

## 🧠 Key Concepts
- **TRNCG Optimization**: Uses a trust region strategy with conjugate gradient steps to efficiently solve sub-problems using matrix-free Hessian-vector products.
- **Matrix-Free Hessian Products**: Avoids explicit computation of the Hessian using nested automatic differentiation.
- **Binary Classification**: Predicts All-Star status based on hitting stats from March through June (when voting closes).
- **Imbalanced Dataset Handling**: BCE loss is weighted to account for class imbalance between All-Stars and non-All-Stars.

---

## 📊 Dataset Construction

- **Seasons**: 2008–2023 (excluding 2020).
- **Features**: 12 input features derived from hitting statistics collected via `pybaseball`.
- **Target**: Binary variable indicating All-Star selection.
- **Sources**:
  - `pybaseball` (Baseball Reference, FanGraphs, etc.)
  - Lahman Baseball Database (All-Star rosters)

**Note**: The cleaned dataset is too large to upload directly. You'll need to download:
- Lahman DB: https://sabr.org/lahman-database/
- `pybaseball`: https://github.com/jldbc/pybaseball

---

## 🛠️ Steps to Reproduce

1. **Data Collection**  
   Use `pybaseball` and the Lahman database to gather and merge hitting statistics and All-Star selection data.

2. **Data Cleaning and Feature Engineering**  
   - Filter stats to include only March–June.
   - Create binary labels for All-Star status.
   - Stratified train/test split.

3. **Model Construction**  
   - 5-layer feedforward neural network (32 → 16 → 8 → 1 neurons).
   - ReLU activations, Sigmoid output.
   - BCEWithLogitsLoss with class weighting.

4. **Optimization via TRNCG**
   - Custom PyTorch-compatible TRNCG optimizer.
   - Uses matrix-free Hessian-vector products via automatic differentiation.
   - Trust region tuning (`δ₀`, `δₘₐₓ`) shown to significantly affect convergence.

5. **Evaluation and Comparison**
   - Accuracy, confusion matrices, and loss convergence plotted.
   - Compared against SGD: TRNCG requires **~10× fewer propagations** to reach the same loss.

---

## 📈 Results Summary

- **TRNCG** outperformed **SGD** in both convergence speed and classification accuracy.
- TRNCG required **4,816 forward / 2,408 backward propagations** to reach a given loss vs. **42,140 each** for SGD.
- Confusion matrices show improved All-Star prediction despite dataset imbalance.
- TRNCG consistently converged within **1 iteration** of Newton CG per epoch.