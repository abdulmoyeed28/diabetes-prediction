# 🩺 Diabetes Prediction Using Machine Learning (SVM)

This project focuses on predicting diabetes based on medical diagnostic data using a **Support Vector Machine (SVM)** classifier. It’s built using the **PIMA Indian Diabetes Dataset**, a popular dataset in healthcare-based machine learning problems.

Rather than manually inputting hypothetical patient values, this implementation **randomly selects a real record from the dataset**, processes it, predicts the outcome, and compares it against the actual result. This simulates real-world testing conditions and ensures that every model run gives a unique, meaningful prediction.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Workflow](#workflow)
- [Model & Evaluation](#model--evaluation)
- [Unique Prediction Mechanism](#unique-prediction-mechanism)
- [How to Run](#how-to-run)
- [Sample Output](#sample-output)
- [Further Improvements](#further-improvements)


---

## 🧠 Overview

This project performs binary classification (diabetic or not) using basic patient health data such as:

- Glucose level
- Blood pressure
- Insulin level
- BMI (Body Mass Index)
- Age
- Pregnancies
- Skin thickness
- Diabetes pedigree function (family history risk)

We train a **Support Vector Machine with a linear kernel**, standardize the data for better accuracy, and evaluate the model's performance using accuracy metrics on both training and testing datasets.

---

## 📂 Dataset

The dataset used is the [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which contains 768 entries and 9 columns:

| Column | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes risk based on family history |
| Age | Age in years |
| Outcome | 0 = Not Diabetic, 1 = Diabetic |

---

## 📦 Libraries Used

- `numpy`
- `pandas`
- `scikit-learn (sklearn)`
  - `StandardScaler`
  - `train_test_split`
  - `svm`
  - `accuracy_score`
- `random`

---

## 🔁 Workflow

1. **Load Dataset**  
   Read the CSV file using Pandas and inspect data shape, distribution, and summary statistics.

2. **Preprocessing**  
   - Split the data into input features (`X`) and labels (`Y`)
   - Standardize the features using `StandardScaler` to bring them to the same scale.

3. **Train/Test Split**  
   - Split data into training and testing sets (80/20 split) using `train_test_split`, stratified by label to preserve class distribution.

4. **Model Training**  
   - Train an **SVM classifier with a linear kernel** using the training data.

5. **Model Evaluation**  
   - Predict outcomes on both the training and test datasets.
   - Measure accuracy using `accuracy_score`.

---

## 🎲 Unique Prediction Mechanism

Instead of asking the user to input new values manually, the project adds a real-world twist by simulating real patient testing.

```python
random_index = random.randint(0, len(diabetes_dataset) - 1)
new_data = diabetes_dataset.drop(columns='Outcome', axis=1).iloc[random_index]
```

- A random patient is selected from the dataset.  
- The input is passed through the same preprocessing pipeline as training data.  
- The model predicts the outcome: Diabetic or Not Diabetic.  
- The prediction is compared with the actual recorded outcome for evaluation.  
- This keeps the testing meaningful and different each time the model runs.

---

## ▶️ How to Run
Make sure you have Python installed and all required packages (numpy, pandas, sklearn).

1. **Clone this repository**
```
git clone https://github.com/your-username/diabetes-prediction-svm.git
cd diabetes-prediction-svm
```

2. **Install dependencies**
```
pip install -r requirements.txt
```
Run the Python script (or Jupyter Notebook)  
```
python diabetes_prediction.py
```
Or open the notebook in Jupyter:  
```
jupyter notebook Diabetes_Prediction_SVM.ipynb
```

## 🖥️ Sample Output  

Accuracy score of the training data :  0.78  
Accuracy score of the test data :  0.77  

Random Index Chosen: 134  
Prediction: Diabetic  
Actual Outcome: Diabetic  
Each time you run the code, a new random patient is selected, giving a fresh and realistic prediction experience.  

## 🚀 Further Improvements
- Add confusion matrix and precision/recall/F1-scores for a deeper evaluation.  
- Experiment with non-linear kernels (RBF, polynomial) or other models like Random Forests or XGBoost.  
- Build a simple web app using Flask or Streamlit to interact with the model.  
- Use cross-validation to avoid overfitting and better validate the model.  
