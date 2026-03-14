
# 🏦 Bank Loan Classification Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Predicting Term Deposit Subscriptions with Machine Learning**

[📊 View Notebook](Loan_classification.ipynb) · [📋 Report Issues](../../issues) · [🤝 Contribute](../../pulls)

</div>

---

## 📋 Table of Contents

- [📖 Overview](#-overview)
- [🎯 Objectives](#-objectives)
- [📁 Dataset](#-dataset)
- [🛠️ Technology Stack](#️-technology-stack)
- [🔧 Installation](#-installation)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [⚙️ Methodology](#️-methodology)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📈 Results & Performance](#-results--performance)
- [📁 Project Structure](#-project-structure)
- [🚀 Usage](#-usage)
- [📊 Key Findings](#-key-findings)
- [🔮 Future Improvements](#-future-improvements)
- [👥 Contributors](#-contributors)
- [📄 License](#-license)

---

## 📖 Overview

This project implements a comprehensive **machine learning pipeline** for predicting whether a client will subscribe to a term deposit at a Portuguese banking institution. The project leverages the renowned **UCI Bank Marketing Dataset** and implements multiple classification algorithms with advanced techniques for handling class imbalance.

### Business Problem
Banks need to efficiently identify potential customers who are likely to subscribe to term deposits. Traditional marketing approaches are costly and time-consuming. This ML solution helps optimize marketing efforts by predicting customer subscription likelihood.

---

## 🎯 Objectives

- ✅ Build a robust classification model to predict term deposit subscriptions
- ✅ Handle class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique)
- ✅ Compare multiple machine learning algorithms
- ✅ Perform comprehensive exploratory data analysis (EDA)
- ✅ Optimize model performance through proper evaluation metrics
- ✅ Create a reproducible and well-documented ML pipeline

---

## 📁 Dataset

**Source:** [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

| Attribute | Description | Type |
|-----------|-------------|------|
| **age** | Client age | Numeric |
| **job** | Type of job | Categorical |
| **marital** | Marital status | Categorical |
| **education** | Education level | Categorical |
| **default** | Has credit in default? | Binary |
| **balance** | Average yearly balance (euros) | Numeric |
| **housing** | Has housing loan? | Binary |
| **loan** | Has personal loan? | Binary |
| **contact** | Contact communication type | Categorical |
| **day** | Last contact day of the month | Numeric |
| **month** | Last contact month | Categorical |
| **duration** | Last contact duration (seconds) | Numeric |
| **campaign** | Number of contacts in this campaign | Numeric |
| **pdays** | Days since last contacted | Numeric |
| **previous** | Contacts before this campaign | Numeric |
| **poutcome** | Outcome of previous campaign | Categorical |
| **y** | **Target**: Term deposit subscription | **Binary** |

**Dataset Statistics:**
- **Total Instances:** 45,211
- **Features:** 16 (15 features + 1 target)
- **Target Distribution:** Imbalanced (requires SMOTE)

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn, Imbalanced-Learn |
| **Models** | Random Forest, Decision Tree, AdaBoost |
| **Environment** | Jupyter Notebook, Google Colab |

---

## 🔧 Installation

### Prerequisites
```bash
python >= 3.8
pip >= 21.0
```

### Install Dependencies
```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Requirements.txt
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
ucimlrepo>=0.0.7
```

---

## 📊 Exploratory Data Analysis

### Key EDA Steps Performed

1. **Data Loading & Inspection**
   - Loaded dataset from UCI Repository
   - Checked data types and missing values
   - Initial statistical summary

2. **Missing Value Analysis**
   - Identified missing values in categorical columns
   - Applied appropriate imputation strategies

3. **Feature Distribution Analysis**
   - Visualized categorical variable distributions
   - Analyzed numerical variable distributions
   - Identified outliers using boxplots

4. **Correlation Analysis**
   - Examined feature correlations
   - Identified multicollinearity issues

5. **Target Variable Analysis**
   - Analyzed class distribution
   - Identified class imbalance requiring SMOTE

### Visualizations Included
- 📊 Histograms for numerical features
- 📊 Bar charts for categorical features
- 📊 Box plots for outlier detection
- 📊 Correlation heatmaps
- 📊 Class distribution charts

---

## ⚙️ Methodology

### Data Preprocessing Pipeline

```python
1. Data Loading
   ↓
2. Missing Value Treatment
   ↓
3. Feature Encoding (One-Hot/Label Encoding)
   ↓
4. Feature Scaling (RobustScaler)
   ↓
5. Train-Test Split
   ↓
6. SMOTE for Class Imbalance
   ↓
7. Model Training & Evaluation
```

### Key Preprocessing Steps

#### 1. Missing Value Handling
```python
# Fill categorical missing values with mode
for col in ['job', 'education']:
    mode_value = Bn[col].mode()[0]
    Bn[col].fillna(mode_value, inplace=True)
```

#### 2. Feature Encoding
```python
# One-hot encoding for categorical variables
Bn_encoded = pd.get_dummies(Bn, drop_first=True)
```

#### 3. Feature Scaling
```python
# RobustScaler to handle outliers
scaler = RobustScaler()
X_train_smote[numerical_cols] = scaler.fit_transform(X_train_smote[numerical_cols])
```

#### 4. Handling Class Imbalance
```python
# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

---

## 🤖 Machine Learning Models

### Models Implemented

| Model | Description | Best Use Case |
|-------|-------------|---------------|
| **Random Forest** | Ensemble of decision trees | General classification, handles overfitting |
| **Decision Tree** | Single tree-based model | Interpretability, simple patterns |
| **AdaBoost** | Boosting ensemble method | Improving weak learners, imbalanced data |

### Model Implementation

```python
# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)

# AdaBoost Classifier
ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
y_pred = ada_classifier.predict(X_test)
```

---

## 📈 Results & Performance

### Model Comparison (With SMOTE)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | 79.2% | 0.85 | 0.91 | 0.88 |
| **Decision Tree** | 72.2% | 0.86 | 0.80 | 0.83 |
| **AdaBoost** | 69.1% | 0.86 | 0.76 | 0.81 |

### Key Performance Insights

✅ **Random Forest** achieved the best overall performance with highest F1-Score (0.88)

✅ **SMOTE** effectively handled class imbalance, improving minority class recall

✅ All models showed good precision (>0.85) indicating low false positive rates

✅ **Recall** improvement is crucial for identifying potential subscribers

### Confusion Matrix Analysis

```
                    Predicted
                  No      Yes
Actual  No      [TN]    [FP]
        Yes     [FN]    [TP]
```

**Key Metrics:**
- **True Positive Rate (Recall):** Ability to identify subscribers
- **Precision:** Accuracy of positive predictions
- **F1-Score:** Harmonic mean of precision and recall

---

## 📁 Project Structure

```
Loan_Classification/
├── Loan_classification.ipynb      # Main Jupyter Notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/
│   └── bank_marketing.csv        # Dataset (if stored locally)
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA Notebook (optional)
├── models/
│   └── saved_models/             # Trained models (optional)
└── results/
    └── performance_metrics.csv   # Model results (optional)
```

---

## 🚀 Usage

### Run the Notebook

```bash
# Open Jupyter Notebook
jupyter notebook Loan_classification.ipynb

# Or use Jupyter Lab
jupyter lab Loan_classification.ipynb
```

### Google Colab

1. Upload `Loan_classification.ipynb` to Google Drive
2. Open with Google Colab
3. Run all cells sequentially

### Custom Dataset

```python
# Load your own dataset
import pandas as pd
df = pd.read_csv('your_dataset.csv')

# Follow the preprocessing pipeline in the notebook
```

---

## 📊 Key Findings

### 🔍 Data Insights

1. **Class Imbalance:** Original dataset showed significant class imbalance, addressed effectively with SMOTE

2. **Important Features:**
   - `duration` (contact duration) - Highly predictive
   - `pdays` (days since last contact) - Significant impact
   - `previous` (previous contacts) - Important feature
   - `balance` (account balance) - Moderate importance

3. **Categorical Patterns:**
   - Certain job types show higher subscription rates
   - Marital status influences subscription likelihood
   - Education level correlates with subscription

### 🎯 Model Insights

1. **Random Forest** outperformed other models due to:
   - Better handling of non-linear relationships
   - Reduced overfitting through ensemble method
   - Better feature importance capture

2. **SMOTE Impact:**
   - Improved recall for minority class
   - Balanced precision-recall trade-off
   - Better overall F1-score

3. **Feature Importance:**
   - Contact duration is the most important predictor
   - Previous campaign outcome is significant
   - Customer demographics play a role

---

## 🔮 Future Improvements

### Short-term
- [ ] Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- [ ] Cross-validation for robust performance estimation
- [ ] Feature engineering (create new features from existing ones)
- [ ] Try additional models (XGBoost, LightGBM, CatBoost)

### Medium-term
- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Create interactive dashboard using Streamlit/Plotly
- [ ] Implement model monitoring and drift detection
- [ ] A/B testing framework for model deployment

### Long-term
- [ ] Real-time prediction pipeline
- [ ] Automated model retraining pipeline
- [ ] Integration with bank's CRM system
- [ ] Explainable AI (SHAP, LIME) for model interpretability

---

## 👥 Contributors

| Contributor | Role | Contact |
|-------------|------|---------|
| **Project Author** | ML Engineer, Data Analysis | [Email](mailto:your-email@example.com) |

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the Bank Marketing dataset
- **Scikit-Learn** community for excellent ML libraries
- **Kaggle** community for insights and best practices
- **Open Source Community** for various tools and libraries used

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **Email:** ziad.elzahar05@gmail.com
- **LinkedIn:** [Ziad Elzahar](www.linkedin.com/in/ziad-elzahar)


<div align="center">

### ⭐ If you found this project helpful, please give it a star!


[🔝 Back to Top](#-bank-loan-classification-project)

</div>
