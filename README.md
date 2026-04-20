# Machine Learning in Extrusion — PCG Defect Prediction

Predicting **Peripheral Coarse Grain (PCG)** defects and **grain size** in Al-Mg-Si aluminium extrusion profiles using supervised machine learning.

> Originally implemented in **MATLAB** as a university project. This repository is a Python / scikit-learn port that replicates the same methodology and dataset, refactored into a modular structure with shared utilities.

---

## Problem

During aluminium extrusion, a defect called **PCG (Peripheral Coarse Grain)** can form on the surface of the profile. It reduces mechanical strength, corrosion resistance, and surface quality — making the component unsafe for automotive structural use.

Grain size is linked to yield strength via the **Hall–Petch relation**:

$$\sigma_y = \sigma_0 + \frac{k}{\sqrt{d}}$$

Predicting PCG presence and grain size from process parameters enables real-time quality control and process optimisation.

---

## Repository Structure

```
ml-extrusion-pcg/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── estrazione.xlsx              # dataset (22 803 FEM samples, 17 columns)
│
├── src/                             # reusable code — shared by all experiments
│   ├── __init__.py
│   ├── config.py                    # paths, column names, RANDOM_STATE
│   ├── data.py                      # loading + preprocessing (train/test split + scaling)
│   ├── metrics.py                   # FNR, FPR, log-loss, MSE, RMSE, MAE, MAPE, R²
│   └── plots.py                     # heatmaps, scatter plots, residual plots
│
├── scripts/                         # one script per experiment
│   ├── 01_classification_neural_network.py
│   ├── 02_classification_logistic_regression.py
│   ├── 03_classification_svm.py
│   ├── 04_regression_neural_network.py
│   ├── 05_regression_linear_model.py
│   └── 06_regression_svr.py
│
└── plots/                           # auto-generated figures
```

The refactoring moved all shared logic (data loading, metric computation, plot styling) into `src/` so each experiment script only contains the logic that is unique to its model. This removes code duplication and makes it easy to add new experiments.

---

## Dataset

`estrazione.xlsx` — 22 803 samples from finite element simulations of Al-Mg-Si extrusion.

| Column | Description |
|---|---|
| `Condition` | Process regime (e.g. LBT-2mm/s, HBT-5mm/s) |
| `Temperature [°C]` | Local temperature |
| `Plastic Strain` | Accumulated plastic deformation |
| `Max Strain Rate` | Maximum strain rate |
| `Effective Stress [MPa]` | Von Mises equivalent stress |
| `Dimensione Grano` | Grain size — **regression target** |
| `PCG (corretti)` | PCG presence (yes/no) — **classification target** |

Five input features were selected based on physical reasoning (temperature drives recrystallisation, plastic strain accumulation triggers PCG, etc.) and exploratory data analysis (PCG frequency histograms per feature interval).

---

## Models

### Classification (PCG presence)

| Script | Model | Key parameters |
|---|---|---|
| `01` | Neural Network (MLP) | Grid: 1–8 neurons × 2 layers, activations: ReLU / Sigmoid / tanh |
| `02` | Logistic Regression | Ridge & Lasso, λ ∈ [10⁻⁶, 10¹] |
| `03` | SVM (RBF) | Grid: C ∈ [0.1, 10], γ ∈ [0.1, 10] |

**Primary metric: FNR (False Negative Rate).** A missed PCG defect means an unsafe component reaches the end user — in this safety-critical context, FNR matters more than overall accuracy.

### Regression (grain size)

> **Physical constraint:** only samples **without PCG** are used. PCG alters microstructural evolution and would contaminate the regression model.

| Script | Model | Key parameters |
|---|---|---|
| `04` | Neural Network (MLP) | Layers: (20, 10), ReLU, α=1e-4 |
| `05` | Linear Regression | OLS |
| `06` | SVR (Gaussian) | C=10, ε=0.5, γ=scale |

Metrics: **MSE, RMSE, MAE, MAPE, R²**

---

## Results

### Classification

For the neural-network classifier, **tanh** gives the most stable performance across all tested architectures. Its cross-entropy loss is uniformly low (~0.05–0.10), while ReLU is noticeably more sensitive to the architecture choice (loss varies from ~0.06 to ~0.18 depending on neuron counts). This is expected for very small networks (1–8 neurons per layer): ReLU's "dying neurons" problem is more impactful when the network has little capacity to absorb lost neurons, whereas tanh is centred around zero and suffers less from this issue.

| Model | FNR | FPR | Stability |
|---|---|---|---|
| **Neural Network (tanh)** | ≈ 0 | ≈ 0 | Best — low and uniform across architectures |
| Logistic Regression | High | Low | Insufficient — poor PCG sensitivity |
| SVM (RBF) | Low | Low | Good but parameter-sensitive |

### Regression

All metrics below come from the original MATLAB implementation on the same 80/20 split.

| Model | MSE | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|---|
| **Neural Network** | 9.9170 | 3.1491 | 2.2554 | 11.18% | **0.8514** |
| SVR | 10.5754 | 3.2520 | 2.1957 | 10.38% | 0.8415 |
| Linear Regression | 49.3886 | 7.0277 | 5.6041 | 28.56% | 0.2262 |

**→ Neural Network is the best model for both tasks.** The Python port may yield slightly different values due to library-specific implementation details (scikit-learn vs MATLAB Statistics and Machine Learning Toolbox).

---

## How to Run

```bash
# Clone
git clone https://github.com/aicezhe/ml-extrusion-pcg.git
cd ml-extrusion-pcg

# Install dependencies
pip install -r requirements.txt

# Place the dataset in data/estrazione.xlsx

# Run any script from the project root
python -m scripts.01_classification_neural_network
python -m scripts.04_regression_neural_network
# ... etc

# All figures are saved to plots/
```

Scripts are run as modules (`python -m scripts.NAME`) so that the `src/` imports resolve correctly.

---

## Key Takeaways

- **FNR is the primary metric** for PCG classification — missing a defect is far more costly than a false alarm.
- **tanh outperforms ReLU on this task** because the network is small (1–8 neurons/layer) — ReLU's dying neurons hurt more when capacity is limited.
- **Regression must be trained on PCG-free samples only** — mixing PCG-positive and PCG-negative microstructures contaminates the model (physically motivated restriction).
- **Neural Network wins on both tasks**, with SVR as a strong runner-up for regression. Linear regression is insufficient.

---

## References

- [Extrusion process — Dassault Systèmes](https://www.3ds.com/it/make/guide/process/extrusion)
- [PCG in Al-Mg-Si extrusion — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1044580324001049)
- [Microstructure evolution in extrusion — OSTI](https://www.osti.gov/biblio/861397/)
