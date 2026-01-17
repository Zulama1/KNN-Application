# 🎬 Netflix Rating Prediction using KNN

This mini-project applies **Machine Learning classification** techniques to predict the **content rating** of Netflix shows and movies based on available metadata. The workflow includes data preprocessing, feature engineering, model training using **K-Nearest Neighbors (KNN)**, and performance evaluation.

---

## 📌 Project Overview

* **Goal:** Predict the *rating* (e.g., TV-MA, PG-13, R, etc.) of Netflix content
* **Dataset:** Netflix Titles Dataset (`netflix_titles.csv`)
* **Model Used:** K-Nearest Neighbors (KNN) Classifier
* **Evaluation Metrics:** Accuracy, Classification Report, Confusion Matrix

This project was completed as an academic assignment and is included here as part of my **ML project portfolio**.

---

## 🧠 Key Concepts Used

* Data Cleaning & Feature Selection
* Handling Missing Values
* Feature Engineering (duration conversion)
* Label Encoding (categorical → numerical)
* Feature Scaling using StandardScaler
* KNN Classification
* Model Evaluation & Visualization

---

## 🗂️ Dataset Description

After preprocessing, the dataset contains the following features:

| Feature         | Description                      |
| --------------- | -------------------------------- |
| `type`          | Movie or TV Show                 |
| `country`       | Country of production            |
| `release_year`  | Year of release                  |
| `duration_mins` | Duration converted to minutes    |
| `rating`        | Target variable (content rating) |

Irrelevant columns such as title, description, director, cast, etc., were removed to reduce noise .

---

## ⚙️ Data Preprocessing

* Missing values filled with `"Unknown"` or `0`
* TV show seasons converted into approximate minutes
* Categorical features encoded using `LabelEncoder`
* Features standardized using `StandardScaler`
* Dataset split into **80% training** and **20% testing**

---

## 🤖 Model Details

* **Algorithm:** K-Nearest Neighbors (KNN)
* **Number of Neighbors:** 16
* **Distance Metric:** Manhattan
* **Algorithm Type:** Brute-force search

```python
KNeighborsClassifier(
    n_neighbors=16,
    metric='manhattan',
    algorithm='brute'
)
```

---

## 📊 Results

* **Accuracy:** ~45.8%
* The model performs reasonably on frequently occurring ratings such as `TV-MA` and `TV-14`
* Lower performance on rare rating classes due to **class imbalance**

### Classification Report & Confusion Matrix

A confusion matrix visualization was generated to analyze misclassifications across rating categories .

---

## 📈 Observations

* Dataset is **highly imbalanced**, affecting prediction accuracy
* KNN struggles with sparse classes
* Encoding ratings as numerical labels introduces ordinal assumptions

---

## 🚀 Future Improvements

* Apply class balancing techniques (SMOTE / undersampling)
* Try alternative models (Random Forest, XGBoost, Logistic Regression)
* Use multi-label or hierarchical classification
* Perform hyperparameter tuning with GridSearchCV
* Improve feature engineering (genre embeddings, text features)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib

---

## 📄 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Run the notebook or script containing the model
