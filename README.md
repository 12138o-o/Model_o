# 🔍 Low Back Pain Risk Prediction Tool

---

An interactive web application based on machine learning for predicting low back pain risk in middle-aged and elderly individuals with abdominal obesity.

---

## 🎯 Project Overview

A **Stacking Classifier** model for low back pain risk prediction, specifically designed for **middle-aged and elderly individuals (≥45 years)** with **abdominal obesity**.

### Key Features

- 🎯 **Target Population**: Middle-aged and elderly (≥45 years) + Abdominal obesity
- 🌍 **Multi-population Support**: Different abdominal obesity standards for USA and China
- 🤖 **Model**: Stacking Classifier (5 base learners + Logistic Regression)
- 📊 **Interpretability**: SHAP analysis showing feature contributions
- 🔬 **Data**: NHANES training (1,582 samples), CHARLS external validation (5,502 samples)

---

## 🚀 Quick Start

### Local Installation
```bash
pip install -r requirements.txt
```

### Local Deployment
```bash
streamlit run app.py
```

The app will automatically open in your browser at: `http://localhost:8501`

---

## 📖 Usage Workflow

### Step 1: Population Screening

Select patient's basic information:

| Item | Options | Description |
|------|---------|-------------|
| **Age Group** | ≥45 years / <45 years | ⚠️ Age <45 cannot proceed |
| **Nationality** | USA / China | Used to determine waist threshold |
| **Gender** | Female / Male | Used to determine waist threshold |

**Important Notes**:
- Age and nationality/gender are **used only for population screening**
- **Do NOT affect prediction probability**
- Prediction is based solely on 5 clinical features

**Waist Circumference Standards**:
- 🇺🇸 USA: Male ≥102cm, Female ≥88cm
- 🇨🇳 China: Male ≥90cm, Female ≥85cm

**Complete Step 1**:
- After filling in, click "✅ Confirm Eligibility and Continue" button
- ⚠️ If age <45, button will be disabled
- Step 2 will only appear after confirmation

### Step 2: Clinical Features Input

**This section only appears after completing Step 1 confirmation!**

Enter the following 5 features:

| Feature | Type | Values |
|---------|------|--------|
| Neck pain | Binary | No(0) / Yes(1) |
| Arthritis | Binary | No(0) / Yes(1) |
| Self-perceived health status | Ordinal | Poor(1) / Fair(2) / Good(3) |
| Lung disease | Binary | No(0) / Yes(1) |
| Waist-circumference | Continuous | Centimeters (cm) |

### Step 3: View Prediction Results

After entering 5 features in Step 2:
- Click "🔮 Start Prediction" to predict
- Click "🔄 Reset" to return to Step 1 and reselect population

**If all conditions are met** (Age ≥45 + Waist meets threshold), displays:

1. **Eligibility Confirmation**
   - ✅ Age: ≥45 years
   - ✅ Waist: Meets abdominal obesity threshold

2. **Prediction Results**
   - Predicted Class: Low Risk / At Risk
   - Risk Probability: 0-100%

3. **SHAP Explanation**
   - Waterfall plot showing each feature's contribution
   - Positive value = increases risk, negative value = decreases risk

**If conditions are not met**:
- ❌ Age <45: Error message, cannot predict
- ❌ Waist below threshold: Warning message, cannot predict

---

## 🧪 Example Scenarios

### Scenario 1: Eligible Middle-aged Patient

```
Step 1 Input:
Age Group: ≥45 years
Nationality: USA
Gender: Female

✅ Click confirm, enter Step 2

Step 2 Input:
Neck pain: Yes (1)
Arthritis: Yes (1)
Self-perceived health status: Fair (2)
Lung disease: No (0)
Waist-circumference: 95.0 cm

🔮 Click "Start Prediction"

Output:
✅ Eligibility Confirmed
   ✓ Age: ≥45 years
   ✓ Waist: 95.0 cm (≥88 cm)
   
Predicted Class: At Risk
Risk Probability: 65.3%
[SHAP Waterfall Plot]
```

### Scenario 2: Age Ineligible (Cannot proceed to Step 2)

```
Step 1 Input:
Age Group: <45 years
Nationality: USA
Gender: Male

Step 1 Output:
⚠️ Warning: Age <45 years selected. 
   You will NOT be able to make predictions.
   
Button Status:
❌ Age <45 - Cannot Continue (disabled)

Result:
Cannot enter Step 2, cannot input clinical features or predict
```

### Scenario 3: Waist Below Threshold (Blocked at Step 2 prediction)

```
Step 1 Input:
Age Group: ≥45 years
Nationality: USA
Gender: Male

✅ Click confirm, enter Step 2

Step 2 Input:
Neck pain: No (0)
Arthritis: No (0)
Self-perceived health status: Good (3)
Lung disease: No (0)
Waist-circumference: 85 cm

🔮 Click "Start Prediction"

Output:
❌ Cannot Make Prediction
⚠️ Waist circumference (85.0 cm) is below 
   the abdominal obesity threshold (102.0 cm) 
   for USA Male.
[No prediction results displayed]
```

---

## 🤖 Model Information

### Model Architecture

**Stacking Classifier**

```
Base Learners:
├─ SVM (Support Vector Machine)
├─ Extra Trees
├─ LightGBM (Light Gradient Boosting Machine)
├─ Decision Tree
└─ LDA (Linear Discriminant Analysis)
      ↓
Meta Learner:
└─ Logistic Regression
```

### Training Strategy

- **Hyperparameter Optimization**: Bayesian Optimization (150 iterations)
- **Cross-validation**: 5-fold repeated stratified cross-validation
- **Optimization Objective**: Maximize AUC
- **Class Balance**: `class_weight='balanced'`

### Datasets

| Dataset | Source | Samples | Purpose |
|---------|--------|---------|---------|
| Training | USA NHANES | 1,582 | Model training |
| Test | USA NHANES | 679 | Internal validation |
| External Validation | China CHARLS | 5,502 | External validation |

---

## 📁 File Structure

```
streamlit_0/
├── app.py                    # Main application
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Data directory
│   ├── X_train.csv          # Training set (scaled)
│   ├── X_train_notscaled.csv
│   ├── X_test.csv           # Test set
│   ├── X_charls_scaled.csv  # External validation (scaled)
│   ├── X_charls_notscaled.csv
│   └── Variable .xlsx       # Variable definitions
├── model/                    # Model directory
│   └── stacking_classifier_model_optimized_v3.pkl
└── .streamlit/               # Streamlit configuration
    └── config.toml
```

---

## 🛠️ Tech Stack

- **Streamlit** `>=1.36` - Web application framework
- **scikit-learn** `>=1.2` - Machine learning
- **SHAP** `>=0.43` - Model interpretability
- **pandas** `>=1.5` - Data processing
- **numpy** `>=1.24` - Numerical computing
- **matplotlib** `>=3.7` - Visualization

---

## ⚙️ Eligibility Requirements

### Must meet the following conditions to obtain prediction results:

1. ✅ **Age**: ≥45 years (middle-aged and elderly)
2. ✅ **Waist**: Meets abdominal obesity threshold
   - 🇺🇸 USA: Male ≥102cm, Female ≥88cm
   - 🇨🇳 China: Male ≥90cm, Female ≥85cm

### Behavior when conditions are not met:

- ❌ Age <45 years:
  - Hide clinical features input
  - Prediction button disabled
  - Display detailed explanation and recommendations

- ❌ Waist below threshold:
  - Display error message
  - Do not display prediction results
  - Provide waist threshold reference

---

## 🔧 FAQ

### Q1: Why can't I see Step 2 clinical features input?

**A**: Step 2 only appears after completing Step 1 and clicking the confirm button. Please ensure:
1. Select age ≥45 years (if <45, button will be disabled)
2. Select nationality and gender
3. Click "✅ Confirm Eligibility and Continue" button
4. Then Step 2 with 5 clinical feature inputs will appear

### Q1.1: How to return to Step 1 to reselect population?

**A**: Click the "🔄 Reset" button in Step 2 to return to Step 1.

### Q2: Do age and nationality affect prediction results?

**A**: **No**. Age and nationality are only used for:
- Screening eligible population (age ≥45)
- Determining waist threshold (different nationality/gender)
- Prediction is based solely on 5 clinical features

### Q3: Why is there no result after clicking predict?

**A**: Check if eligibility requirements are met:
- ✓ Age ≥45 years
- ✓ Waist meets threshold

### Q4: What if SHAP calculation is slow?

**A**: First calculation will be cached, subsequent predictions will be much faster (optimized to max 1000 background samples).

### Q5: Are predictions reliable for Chinese population?

**A**: Model is trained on USA data, predictions for Chinese population are for reference only. App will display applicability notes.

---

## ⚠️ Disclaimer

1. **Clinical Decision**: This tool is for reference only, **cannot replace professional medical advice**

2. **Applicable Scope**:
   - ✅ Middle-aged and elderly (≥45 years)
   - ✅ Individuals with abdominal obesity
   - ❌ Young adults (<45 years)
   - ❌ Special populations (pregnant women, severe illness patients)

3. **Model Limitations**:
   - Trained on USA population
   - Prediction accuracy for Chinese population may be reduced

4. **Data Privacy**:
   - All computations completed locally
   - Does not store or upload any data

---

## 📚 Resources

### Data Sources
- [NHANES](https://www.cdc.gov/nchs/nhanes/) - National Health and Nutrition Examination Survey
- [CHARLS](http://charls.pku.edu.cn/) - China Health and Retirement Longitudinal Study

### Technical Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

## 📞 Support

- Check documentation for update history
- Submit issues to report problems
- Refer to related documentation for help

---

**Version**: v3.1  
**Updated**: October 2025  
**Status**: ✅ Actively Maintained

**Quick Start**:
```bash
pip install -r requirements.txt
streamlit run app.py
```

🎉 Enjoy!
