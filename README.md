# 🍷 Wine Quality Prediction — Multiclass Classification

A machine learning project that predicts the quality category (**Low / Medium / High**) of red wine from its physicochemical properties, using and comparing three classification algorithms: **Random Forest**, **SGD Classifier**, and **Support Vector Classifier (SVC)**.

---

## 📑 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Solution Overview](#-solution-overview)
3. [Dataset](#-dataset)
4. [Data Quality Issues](#-data-quality-issues)
5. [Exploratory Data Analysis](#-exploratory-data-analysis)
6. [Feature Engineering](#-feature-engineering)
7. [Project Pipeline](#-project-pipeline)
8. [Models Used](#-models-used)
9. [Model Evaluation](#-model-evaluation)
   - [Accuracy Comparison](#accuracy-comparison)
   - [Classification Reports](#classification-reports)
   - [Confusion Matrices](#confusion-matrices)
10. [Feature Importance](#-feature-importance)
11. [Final Results & Model Selection](#-final-results--model-selection)
12. [Project Structure](#-project-structure)
13. [How to Run](#-how-to-run)
14. [Requirements](#-requirements)
15. [Limitations & Future Work](#-limitations--future-work)
16. [Author](#-author)

---

## 🎯 Problem Statement

Wine quality is traditionally assessed by human tasters, which is subjective, expensive, and hard to scale. Each wine in the dataset has measurable **chemical/physicochemical properties** (acidity, sugar, sulfur dioxide, density, pH, alcohol, etc.) along with a **quality score** assigned by tasters.

> **Can we use the chemical properties of a wine to predict its quality?**

This project frames that question as a supervised machine learning problem.

---

## 💡 Solution Overview

The raw `quality` column contains scores from 3–8, but the classes are **heavily imbalanced** (most wines score 5 or 6; almost none score 3 or 8). Training a 6-class classifier directly would give the model very few examples to learn the rare classes.

**Solution:** the quality scores are grouped into three broader categories:

| Original Score | New Category |
|---|---|
| 3, 4 | **Low** |
| 5, 6 | **Medium** |
| 7, 8 | **High** |

The final objective becomes:

> **Build a multiclass classification system that predicts Low, Medium, or High wine quality from physicochemical properties, train three different classifiers, evaluate their performance, and identify the most suitable model.**

Three algorithms are trained and compared: **Random Forest**, **SGD Classifier**, and **SVC**, evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrices — not accuracy alone, because of the class imbalance.

---

## 📊 Dataset

**Source file:** `winequality-red.csv` ([UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) — red wine variant)

| Property | Value |
|---|---|
| Rows (raw) | 1,599 |
| Columns | 12 (11 features + 1 target) |
| Missing values | 0 |
| Duplicate rows | 240 |
| Rows after cleaning | 1,359 |

**Features:**

| # | Feature | Meaning |
|---|---|---|
| 1 | Fixed acidity | Non-volatile acids — affects freshness/sourness |
| 2 | Volatile acidity | Acetic-acid-related acidity — vinegar-like taste if high |
| 3 | Citric acid | Contributes freshness/citrus character |
| 4 | Residual sugar | Sugar left after fermentation — sweetness |
| 5 | Chlorides | Salt-related content |
| 6 | Free sulfur dioxide | Available SO₂ — protects against oxidation |
| 7 | Total sulfur dioxide | Free + bound SO₂ |
| 8 | Density | Related to sugar/alcohol composition |
| 9 | pH | Acidity/basicity measure |
| 10 | Sulphates | Sulfate-related compounds — preservation |
| 11 | Alcohol | % alcohol by volume — strength/body |
| 12 | **Quality** (target) | Taster-assigned score (3–8), later grouped into Low/Medium/High |

**Sample rows:**

| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | ... | alcohol | quality |
|---|---|---|---|---|---|---|---|
| 7.4 | 0.70 | 0.00 | 1.9 | 0.076 | ... | 9.4 | 5 |
| 7.8 | 0.88 | 0.00 | 2.6 | 0.098 | ... | 9.8 | 5 |
| 7.8 | 0.76 | 0.04 | 2.3 | 0.092 | ... | 9.8 | 5 |
| 11.2 | 0.28 | 0.56 | 1.9 | 0.075 | ... | 9.8 | 6 |

**Summary statistics (post-cleaning, `df.describe()`):**

| Stat | fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free SO₂ | total SO₂ | density | pH | sulphates | alcohol | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 8.31 | 0.53 | 0.27 | 2.52 | 0.088 | 15.89 | 46.83 | 0.9967 | 3.31 | 0.66 | 10.43 | 5.62 |
| std | 1.74 | 0.18 | 0.20 | 1.35 | 0.049 | 10.45 | 33.41 | 0.0019 | 0.16 | 0.17 | 1.08 | 0.82 |
| min | 4.60 | 0.12 | 0.00 | 0.90 | 0.012 | 1 | 6 | 0.9901 | 2.74 | 0.33 | 8.40 | 3 |
| max | 15.90 | 1.58 | 1.00 | 15.50 | 0.611 | 72 | 289 | 1.0037 | 4.01 | 2.00 | 14.90 | 8 |

---

## ⚠️ Data Quality Issues

The dataset was checked for common quality problems before modeling:

- **Missing values:** None found across all 12 columns (`df.isnull().sum()` → all zero).
- **Duplicate rows:** 240 exact duplicate rows were found and removed, reducing the dataset from 1,599 → **1,359** rows.
- **Class imbalance in `quality`:** Scores are concentrated around 5–6, with very few wines at the extremes (3 or 8):

  | Quality | Approx. Count |
  |---|---|
  | 3 | ~10 |
  | 4 | ~50 |
  | 5 | ~575 |
  | 6 | ~535 |
  | 7 | ~165 |
  | 8 | ~15 |

  This imbalance is the primary motivation for grouping scores into **Low / Medium / High**, and for evaluating models using Precision/Recall/F1 rather than accuracy alone.
- **Outliers:** Several features (residual sugar, chlorides, total SO₂, sulphates) are strongly right-skewed with long tails, indicating a small number of wines with unusually extreme values (see histograms below). These were kept rather than removed, since tree- and margin-based models can be reasonably robust to them and removing data would worsen the existing class imbalance.

---

## 🔎 Exploratory Data Analysis

### Quality Score Distribution

![Quality Distribution](./assets/quality_distribution.png)

Most wines have quality scores of **5 or 6**; scores of 3 and 8 are rare — confirming the class imbalance in the raw target.

### Feature Distributions

![Feature Histograms](./assets/feature_histograms.png)

| Feature | Shape | Interpretation |
|---|---|---|
| Fixed acidity | Right-skewed | Most values 7–9, long tail to ~16 |
| Volatile acidity | Right-skewed | Most values 0.3–0.7 |
| Citric acid | Skewed, many zeros | Zero is a very common value |
| Residual sugar | Strongly right-skewed | Most 1–3, few up to 15+ (likely outliers) |
| Chlorides | Strongly right-skewed | Most 0.05–0.12, few up to 0.6 |
| Free SO₂ | Right-skewed | Most 5–25, tail to ~70 |
| Total SO₂ | Strongly right-skewed | Most 0–70, tail to 250+ |
| Density | ~Bell-shaped | Concentrated 0.996–0.998 |
| pH | ~Normal | Concentrated 3.2–3.4 |
| Sulphates | Right-skewed | Most 0.4–0.8 |
| Alcohol | Right-skewed | Most 9–11%, tail to 15% |

### Correlation Heatmap

![Correlation Heatmap](./assets/correlation_heatmap.png)

**Reading the scale:** values range from **-1** (strong negative) to **+1** (strong positive); values near **0** indicate little to no linear relationship.

**Correlation of each feature with `quality`:**

| Feature | Correlation w/ Quality | Interpretation |
|---|---|---|
| **Alcohol** | **+0.48** | Strongest positive relationship — higher alcohol tends to associate with higher quality |
| Sulphates | +0.25 | Weak positive |
| Citric acid | +0.23 | Weak positive |
| Fixed acidity | +0.12 | Very weak positive |
| Residual sugar | +0.01 | ~No linear relationship |
| Free sulfur dioxide | -0.05 | ~No linear relationship |
| pH | -0.06 | ~No linear relationship |
| Chlorides | -0.13 | Very weak negative |
| Total sulfur dioxide | -0.18 | Weak negative |
| Density | -0.18 | Weak negative |
| **Volatile acidity** | **-0.40** | Strongest negative relationship — higher volatile acidity associates with lower quality |

**Notable feature–feature correlations:** fixed acidity ↔ citric acid (+0.67), fixed acidity ↔ pH (-0.69), free SO₂ ↔ total SO₂ (+0.67, expected since total = free + bound), density ↔ alcohol (-0.50).

> **Note:** Correlation measures only *linear* relationships. Weakly-correlated features were **not** dropped, since a machine learning model can capture nonlinear relationships and feature interactions that a simple correlation coefficient cannot.

---

## 🛠 Feature Engineering

The `quality` score was mapped into three balanced-enough categories using:

```python
def quality_category(quality):
    if quality <= 4:
        return "Low"
    elif quality <= 6:
        return "Medium"
    else:
        return "High"
```

**Resulting class distribution:**

![Quality Category Distribution](./assets/quality_category_distribution.png)

| Category | Count | % of Data |
|---|---|---|
| Medium | 1,112 | 81.8% |
| High | 184 | 13.5% |
| Low | 63 | 4.6% |

Even after grouping, **Medium dominates the dataset (~82%)** — this residual imbalance is important context for interpreting every metric below.

---

## 🔄 Project Pipeline

```
Raw CSV (winequality-red.csv, 1599 rows)
        ↓
Data Cleaning (remove 240 duplicate rows → 1359 rows)
        ↓
EDA (histograms, correlation heatmap, class distribution)
        ↓
Feature Engineering (quality → Low / Medium / High)
        ↓
Train/Test Split (80/20, stratified by class)
   → Train: 1087 rows   |   Test: 272 rows
        ↓
Feature Scaling (StandardScaler — fit on train, applied to test)
        ↓
   ┌──────────────┬──────────────┬──────────────┐
   ↓              ↓              ↓
Random Forest   SGD Classifier   SVC
(raw features)  (scaled)         (scaled)
   ↓              ↓              ↓
   └──────────────┴──────────────┘
        ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
        ↓
Model Comparison & Selection
        ↓
Feature Importance Analysis (Random Forest)
        ↓
Inference on New Data
```

**Train/test split** was stratified to preserve class proportions in both sets:

| Set | Medium | High | Low |
|---|---|---|---|
| Train (n=1087) | 81.9% | 13.5% | 4.6% |
| Test (n=272) | 81.6% | 13.6% | 4.8% |

**Feature scaling:** Random Forest does not require scaling (it splits on thresholds, unaffected by scale). SGD and SVC are distance/gradient-based and **do** require scaled features, so `StandardScaler` (fit on training data only, then applied to test data) was used for those two models.

---

## 🤖 Models Used

| Model | Type | Key Idea | Scaling Required |
|---|---|---|---|
| **Random Forest** | Ensemble of decision trees | Builds many trees on random subsets of data/features and combines predictions by majority vote | No |
| **SGD Classifier** | Linear model, gradient-based | Iteratively adjusts model weights via Stochastic Gradient Descent to minimize prediction error | Yes |
| **SVC (Support Vector Classifier)** | Margin-based (RBF kernel) | Finds a decision boundary that maximizes the margin between classes, using the closest points ("support vectors") | Yes |

**Configuration used:**

```python
rf  = RandomForestClassifier(n_estimators=200, random_state=42)
sgd = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
svc = SVC(kernel="rbf", random_state=42)
```

---

## 📈 Model Evaluation

### Accuracy Comparison

| Model | Accuracy |
|---|---|
| Random Forest | 81.99% |
| SGD Classifier | 80.51% |
| **SVC** | **84.56%** |

> ⚠️ **Accuracy alone is misleading here.** Because Medium wines make up ~82% of the test set, a model can score high accuracy simply by predicting "Medium" often — while still failing badly on Low and High. This is exactly what happens with SVC below (0% recall on Low). Precision, Recall, and F1-score per class must be considered.

### Classification Reports

Metric definitions used below:

- **Precision** — of everything the model *predicted* as this class, what fraction was actually correct? (Low precision = many false alarms.)
- **Recall** — of everything that *actually is* this class, what fraction did the model correctly catch? (Low recall = many missed cases.)
- **F1-score** — harmonic mean of precision and recall; a single balanced score, most informative when classes are imbalanced.
- **Support** — the number of actual (true) samples of that class in the test set — this is *not* a performance metric, it's the sample count each score is computed over. Low support (e.g., Low = 13) means that class's metrics are based on very few examples and are statistically less reliable.

**Random Forest**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| High | 0.57 | 0.32 | 0.41 | 37 |
| Low | 0.33 | 0.15 | 0.21 | 13 |
| Medium | 0.85 | 0.94 | 0.90 | 222 |
| **Accuracy** | | | **0.82** | **272** |
| Macro avg | 0.59 | 0.47 | 0.51 | 272 |
| Weighted avg | 0.79 | 0.82 | 0.80 | 272 |

*Meaning:* Random Forest is strong on Medium (F1 = 0.90, the dominant class) but weak on Low (F1 = 0.21) and moderate on High (F1 = 0.41) — it correctly identifies most Medium wines but misses the majority of Low and High wines.

**SGD Classifier**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| High | 0.44 | 0.41 | 0.42 | 37 |
| Low | 1.00 | 0.08 | 0.14 | 13 |
| Medium | 0.86 | 0.91 | 0.88 | 222 |
| **Accuracy** | | | **0.81** | **272** |
| Macro avg | 0.77 | 0.47 | 0.48 | 272 |
| Weighted avg | 0.81 | 0.81 | 0.79 | 272 |

*Meaning:* Low precision is a perfect 1.00, but recall is only 0.08 — meaning that on the rare occasions SGD predicts "Low," it's always correct, but it predicts "Low" for almost none of the actual Low wines (misses 92% of them). SGD achieves the best High-class F1 (0.42) of the three models.

**SVC**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| High | 0.90 | 0.24 | 0.38 | 37 |
| Low | 0.00 | 0.00 | 0.00 | 13 |
| Medium | 0.84 | 1.00 | 0.91 | 222 |
| **Accuracy** | | | **0.85** | **272** |
| Macro avg | 0.58 | 0.41 | 0.43 | 272 |
| Weighted avg | 0.81 | 0.85 | 0.80 | 272 |

*Meaning:* SVC gets the highest accuracy and the best Medium-class F1 (0.91, near-perfect recall) — but it **completely fails on Low** (precision, recall, and F1 all 0.00): it never once predicts "Low" correctly. This is the clearest example in this project of why accuracy alone is not a reliable metric under class imbalance.

### Confusion Matrices

Rows = actual class, columns = predicted class. Diagonal = correct predictions; off-diagonal = errors.

**Random Forest**

![RF Confusion Matrix](./assets/rf_confusion_matrix.png)

| Actual \ Predicted | Low | Medium | High |
|---|---|---|---|
| **Low** (13) | 2 | 11 | 0 |
| **Medium** (222) | 4 | 209 | 9 |
| **High** (37) | 0 | 25 | 12 |

Correctly identifies 209/222 Medium wines, but only 2/13 Low and 12/37 High. The dominant error pattern: 11 Low wines and 25 High wines misclassified as Medium.

**SGD Classifier**

![SGD Confusion Matrix](./assets/sgd_confusion_matrix.png)

| Actual \ Predicted | Low | Medium | High |
|---|---|---|---|
| **Low** (13) | 1 | 12 | 0 |
| **Medium** (222) | 0 | 203 | 19 |
| **High** (37) | 0 | 22 | 15 |

Best of the three models at identifying High wines (15/37 correct), but still confuses most Low wines with Medium (12/13).

**SVC**

![SVC Confusion Matrix](./assets/svc_confusion_matrix.png)

| Actual \ Predicted | Low | Medium | High |
|---|---|---|---|
| **Low** (13) | 0 | 13 | 0 |
| **Medium** (222) | 0 | 221 | 1 |
| **High** (37) | 0 | 28 | 9 |

Identifies 221/222 Medium wines almost perfectly, but **zero** Low wines correctly — every single Low wine is predicted as Medium.

**Common pattern across all three models:** Low and High wines are frequently misclassified as Medium. This is driven by the class imbalance — with far more Medium training examples, all three models learn that "Medium" is a statistically safe default prediction when uncertain.

---

## ⭐ Feature Importance

Extracted from the trained Random Forest (`rf.feature_importances_`):

![Feature Importance](./assets/feature_importance.png)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Alcohol | 0.152 |
| 2 | Volatile acidity | 0.117 |
| 3 | Sulphates | 0.113 |
| 4 | Density | 0.091 |
| 5 | Total sulfur dioxide | 0.089 |
| 6 | Citric acid | 0.082 |
| 7 | Fixed acidity | 0.079 |
| 8 | Residual sugar | 0.076 |
| 9 | Chlorides | 0.070 |
| 10 | pH | 0.068 |
| 11 | Free sulfur dioxide | 0.062 |

**Alcohol, volatile acidity, and sulphates** are the top three drivers of the model's predictions — consistent with the correlation analysis, where alcohol and volatile acidity also had the strongest linear relationships with quality.

---

## 🏆 Final Results & Model Selection

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 Score (weighted) |
|---|---|---|---|---|
| **SVC** | **0.8456** | 0.8109 | 0.8456 | **0.7974** |
| Random Forest | 0.8199 | 0.7899 | 0.8199 | 0.7969 |
| SGD | 0.8051 | 0.8069 | 0.8051 | 0.7862 |

**SVC achieves the highest overall accuracy (84.6%) and the highest weighted F1-score**, driven almost entirely by its near-perfect performance on the Medium class. However, its weighted F1 is only marginally ahead of Random Forest (0.7974 vs 0.7969), and it completely fails to detect any Low-quality wine.

**Recommendation:** if the goal is overall accuracy on a Medium-dominated dataset, **SVC** is the top performer. If correctly identifying **minority classes (Low, High) matters more** — which is arguably the more useful business case, since flagging poor- or excellent-quality wine is often the actual point of quality prediction — **Random Forest** is the more balanced choice: it has non-zero recall on Low (0.15 vs SVC's 0.00) and provides interpretable feature importances. **SGD** is competitive on High recall but is the weakest on Medium and overall F1.

> In short: no single metric tells the whole story here. The model choice should depend on which class matters most for the intended use case, and future work (below) should focus on improving minority-class performance rather than chasing overall accuracy.

---

## 📁 Project Structure

```
wine-quality-prediction/
├── README.md
├── winequality-red.csv                     # Raw dataset
├── wine__main__file.ipynb                  # Main pipeline: EDA → preprocessing → training → evaluation
├── problem_undersatnding.ipynb             # Problem framing & why quality is grouped into 3 classes
├── understaning_dataset.ipynb              # Feature distributions + Random Forest / SVC / SGD deep-dive notes
├── Correlation_Heatmap_Analysis.ipynb      # Correlation heatmap interpretation
├── confusion_matrix_interpretation.ipynb   # Confusion matrix walkthrough for all 3 models
├── feature_s_or__attribute.ipynb           # Feature reference / glossary
├── importan_terms_and_concepts.ipynb       # ML concept notes (histograms, correlation, etc.)
└── assets/                                 # Exported chart images used in this README
    ├── quality_distribution.png
    ├── feature_histograms.png
    ├── correlation_heatmap.png
    ├── quality_category_distribution.png
    ├── rf_confusion_matrix.png
    ├── sgd_confusion_matrix.png
    ├── svc_confusion_matrix.png
    └── feature_importance.png
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd wine-quality-prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the main notebook**
   ```bash
   jupyter notebook wine__main__file.ipynb
   ```
   Run all cells top to bottom — this loads the data, cleans it, performs EDA, trains all three models, and prints/plots all evaluation results shown in this README.

5. **Predict on a new wine (example, from the notebook):**
   ```python
   new_wine = pd.DataFrame({
       "fixed acidity": [7.5], "volatile acidity": [0.50], "citric acid": [0.30],
       "residual sugar": [2.0], "chlorides": [0.08], "free sulfur dioxide": [15],
       "total sulfur dioxide": [40], "density": [0.996], "pH": [3.30],
       "sulphates": [0.65], "alcohol": [10.5]
   })
   rf.predict(new_wine)   # → 'Medium'
   ```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🚧 Limitations & Future Work

- **Class imbalance** is the main bottleneck — all three models struggle on the Low class (only 63 samples total). Future work: try `class_weight="balanced"`, SMOTE/oversampling, or collecting more Low/High samples.
- **Hyperparameter tuning** was not performed (default/lightly-set parameters were used for all three models); `GridSearchCV`/`RandomizedSearchCV` could likely improve minority-class recall.
- **Correlation-only feature screening** was avoided in favor of keeping all 11 features, but formal feature selection (e.g., recursive feature elimination) was not tested.
- **Only 3 algorithms compared** — gradient boosting methods (XGBoost, LightGBM) or a properly tuned SVC with `class_weight` could be explored next.

---

## 👤 Author

*Add your name, GitHub profile, and contact/LinkedIn here.*
