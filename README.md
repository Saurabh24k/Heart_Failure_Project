# Heart Disease Prediction Project

This project aims to predict the presence of heart disease in patients using various machine learning models and a neural network. By analyzing a dataset of patient records, we utilize a range of predictive modeling techniques to identify patterns and characteristics that indicate heart disease risk. This README outlines the project's workflow, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and hyperparameter tuning.

## Project Overview

- **Objective**: To predict heart disease in patients using machine learning and deep learning approaches.
- **Dataset**: The dataset, referred to as `heart.csv`, consists of patient records with various attributes related to heart health.
- **Tools and Libraries**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Keras.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Python packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn keras
  ```

### Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/Saurabh24k/Heart_Failure_Project.git
cd Heart_Failure_Project
```

### Dataset Description

The dataset heart.csv includes several medical and demographic attributes. Key features include age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, and others. The target variable is HeartDisease, indicating the presence (1) or absence (0) of heart disease.

### Data Preprocessing

- Encoding Categorical Variables: Categorical features such as Sex, ChestPainType, RestingECG, ExerciseAngina, and ST_Slope are encoded into numerical values using LabelEncoder.
- Splitting the Dataset: The data is divided into features (X) and the target variable (y). It is then split into training (80%) and testing (20%) sets.

### Exploratory Data Analysis (EDA)

- Visualizations: The distribution of the Age feature is visualized using a histogram to understand the age profile of the patients.

### Model Training

- Logistic Regression: Trained with max_iter=1000.
- Random Forest Classifier: Default parameters.
- Gradient Boosting Classifier: Default parameters.
- Neural Network: A simple architecture with one hidden layer of 10 neurons and an output layer with a sigmoid activation function, optimized using binary crossentropy loss and the adam optimizer, trained over 100 epochs.

### Model Evaluation
- Evaluation metrics include precision, recall, f1-score, and accuracy. The Random Forest Classifier showed the highest accuracy before hyperparameter tuning.
A bar plot comparing the accuracies of the different models is generated.
- Results
  ```bash
  Logistic Regression:
               precision    recall  f1-score   support

           0       0.77      0.88      0.82        77
           1       0.91      0.81      0.86       107

    accuracy                           0.84       184
   macro avg       0.84      0.85      0.84       184
  weighted avg       0.85      0.84      0.84       184

  Random Forest Classifier:
               precision    recall  f1-score   support

           0       0.85      0.90      0.87        77
           1       0.92      0.89      0.90       107

    accuracy                           0.89       184
   macro avg       0.89      0.89      0.89       184
  weighted avg       0.89      0.89      0.89       184

  Gradient Boosting Classifier:
               precision    recall  f1-score   support

           0       0.82      0.90      0.86        77
           1       0.92      0.86      0.89       107

    accuracy                           0.88       184
   macro avg       0.87      0.88      0.87       184
  weighted avg       0.88      0.88      0.88       184

  Neural Network:
               precision    recall  f1-score   support

           0       0.78      0.83      0.81        77
           1       0.87      0.83      0.85       107

    accuracy                           0.83       184
   macro avg       0.83      0.83      0.83       184
  weighted avg       0.83      0.83      0.83       184
  ```

### Hyperparameter Tuning
Conducted for the Random Forest model using GridSearchCV with parameters for n_estimators, max_features, and max_depth.

### Results and Conclusion
The Random Forest Classifier exhibited the highest accuracy among the initial models. Following hyperparameter tuning, the performance of the Random Forest model was further improved, showcasing the effectiveness of ensemble methods in handling complex datasets for binary classification problems like heart disease prediction.
