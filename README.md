# 📱 Phone Addiction Level Prediction

A machine learning project that predicts a user's **Phone Addiction Level** from personal habits, phone usage patterns, lifestyle factors, and related behavioral metrics.

The project includes:
- Data preprocessing and exploratory setup
- Multiple regression models
- Hyperparameter tuning with `GridSearchCV`
- Model evaluation using MAE, RMSE, and R²
- Model serialization with `joblib`
- A Streamlit web application for interactive prediction

---

## 📌 Project Overview

The target variable is:

```text
Addiction_Level
```

The uploaded dataset contains **6,000 records and 20 columns**. The model-training code removes `Name` and `Location` when those columns are present and uses the remaining input variables to predict `Addiction_Level`.

The target values in the uploaded dataset range from **1.0 to 10.0**.

---

## 🎯 Objective

The main objective is to build a regression-based machine learning system that estimates a user's phone addiction level based on factors such as:

- Age
- Gender
- Daily phone usage
- Sleep hours
- Social media usage
- Gaming time
- Education time
- Phone checks per day
- Apps used daily
- Weekend phone usage
- Exercise
- Anxiety
- Depression
- Self-esteem
- Family communication
- Social interactions
- Intellectual performance
- Screen time before bed
- Phone usage purpose

---

## 🗂️ Dataset

The dataset contains the following columns:

| Column | Description |
|---|---|
| `Age` | User age |
| `Gender` | User gender |
| `Daily_Usage_Hours` | Daily phone usage in hours |
| `Sleep_Hours` | Sleep duration |
| `Interllectual_Performance` | Intellectual performance score |
| `Social_Interactions` | Social interaction score |
| `Exercise_Hours` | Exercise duration |
| `Anxiety_Level` | Anxiety score |
| `Depression_Level` | Depression score |
| `Self_Esteem` | Self-esteem score |
| `Screen_Time_Before_Bed` | Screen usage before sleeping |
| `Phone_Checks_Per_Day` | Number of phone checks per day |
| `Apps_Used_Daily` | Number of apps used daily |
| `Time_on_Social_Media` | Social media usage time |
| `Time_on_Gaming` | Gaming time |
| `Time_on_Education` | Education-related phone usage |
| `Phone_Usage_Purpose` | Main phone usage purpose |
| `Family_Communication` | Family communication score |
| `Weekend_Usage_Hours` | Weekend phone usage |
| `Addiction_Level` | Target variable |

> **Note:** The project code uses the column name `Interllectual_Performance` exactly as it appears in the dataset.

---

## 🧠 Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Numerical & Categorical Feature Detection
   ↓
Data Preprocessing
   ↓
Machine Learning Pipeline
   ↓
GridSearchCV Hyperparameter Tuning
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Application
```

---

## 🔧 Data Preprocessing

The training code uses a `ColumnTransformer`:

### Numerical features
Numerical columns are scaled using:

```python
RobustScaler()
```

### Categorical features
Categorical columns are encoded using:

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

The preprocessing and model are combined into a single Scikit-learn `Pipeline`.

This allows the same preprocessing steps to be applied consistently during training and prediction.

---

## 🤖 Models Used

The training notebook/script experiments with several regression algorithms:

### 1. Decision Tree Regressor
Hyperparameters tuned:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### 2. K-Nearest Neighbors Regressor

Hyperparameters tuned:

- `n_neighbors`
- `weights`
- `p`

### 3. Support Vector Regressor (SVR)

Hyperparameters tuned:

- `C`
- `kernel`
- `gamma`
- `epsilon`

### 4. Bagging Regressor

Hyperparameters tuned:

- `n_estimators`
- `max_samples`
- `max_features`
- `bootstrap`

### 5. AdaBoost Regressor

Hyperparameters tuned:

- `n_estimators`
- `learning_rate`
- `loss`

### 6. Gradient Boosting Regressor

The project tests parameters including:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### 7. XGBoost Regressor

The project tests parameters including:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`

---

## ⚙️ Hyperparameter Tuning

The project uses:

```python
GridSearchCV(
    scoring="r2",
    cv=5
)
```

This performs 5-fold cross-validation and searches through the specified parameter combinations.

The best estimator is selected using the R² scoring metric.

---

## 📊 Model Evaluation

The project uses the following regression metrics:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

```text
Lower MAE → smaller average prediction error
```

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

```text
Lower RMSE → smaller prediction error
```

### R² — R-Squared

Measures how much of the variation in the target is explained by the model.

```text
Higher R² → better fit to the evaluation data
```

The code evaluates the models using:

```python
mean_absolute_error()
mean_squared_error()
r2_score()
```

---

## 💾 Model Saving

The training code saves the best estimator using `joblib`:

```python
best_model = grid_search_pipeline.best_estimator_

joblib.dump(
    best_model,
    "phone_addiction_pipeline.pkl"
)
```

The intended saved artifact is therefore a complete Scikit-learn pipeline containing preprocessing and the trained regression model.

---

## 🌐 Streamlit Application

The Streamlit application provides an interactive interface called:

**📱 Phone Addiction Level Predictor**

Users can enter their:

- Personal information
- Phone usage habits
- Lifestyle information
- Behavioral scores

and click:

**🔮 Predict Addiction Level**

The application then displays the predicted value on a scale of:

```text
0 – 10
```

The application also displays an interpretation:

- Below 3.5 → Low
- 3.5 to below 6.5 → Moderate
- 6.5 and above → High

These thresholds are the thresholds implemented in the supplied Streamlit code.

---

## 🖥️ Streamlit Input Features

The web application accepts:

```text
Age
Gender
Daily Usage Hours
Sleep Hours
Time on Social Media
Time on Gaming
Time on Education
Usage Purpose
Phone Checks Per Day
Screen Time Before Bed
Apps Used Daily
Weekend Usage Hours
Intellectual Performance
Social Interactions Score
Exercise Hours
Anxiety Level
Depression Level
Self Esteem Level
Family Communication Score
```

---

## 📁 Recommended GitHub Repository Structure

```text
phone-addiction-prediction/
│
├── app.py
├── phone_addiction_pipeline.pkl
├── phone_addiction_model.py
├── phone_addiction_dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

If the dataset is not intended to be uploaded to GitHub, it can instead be kept locally or provided through another data source.

---

## 📦 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd phone-addiction-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit xgboost
```

---

## ▶️ Run the Streamlit Application

After placing the trained pipeline file in the project directory:

```text
phone_addiction_pipeline.pkl
```

run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🧪 Training the Model

The supplied training code reads the dataset using:

```python
df = pd.read_csv("/content/drive/MyDrive/phone_addiction_dataset.csv")
```

For a local GitHub project, update this path to your local dataset location, for example:

```python
df = pd.read_csv("phone_addiction_dataset.csv")
```

Then run the training script/notebook.

---


## 🔍 Project Files

### `phone_addiction_model.py`
Contains the machine learning workflow, including:

- Dataset loading
- Feature and target separation
- Train-test split
- Numerical/categorical preprocessing
- Regression pipelines
- GridSearchCV
- Model evaluation
- Model saving

### `app.py`
Contains the Streamlit user interface and loads the saved model for prediction.

### `phone_addiction_pipeline.pkl`
Intended to store the trained preprocessing + regression pipeline.

### `phone_addiction_dataset.csv`
Dataset used for model development.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **Streamlit**

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

- Data preprocessing
- Numerical feature scaling
- Categorical encoding
- Train-test splitting
- Machine learning pipelines
- Regression
- Cross-validation
- Hyperparameter tuning
- GridSearchCV
- Model evaluation
- Model serialization
- Streamlit deployment

---

## 🚀 Future Improvements

Possible improvements include:

- Add exploratory data analysis visualizations to the project
- Compare all trained models in one results table
- Add model performance visualizations
- Add input validation for related time-based features
- Improve the Streamlit interface
- Add feature importance/explainability
- Deploy the Streamlit application online
- Add automated model retraining
- Add a requirements file with pinned package versions

---

## 👨‍💻 Author

**Wazid**

MSc Computer Science

Interested in Data Analytics, Data Science, Machine Learning, and Python-based applications.

---

## ⭐ Project Purpose

This project was developed as a practical machine learning application to understand how behavioral and phone-usage features can be used to build a regression model and expose its predictions through a simple web interface.

