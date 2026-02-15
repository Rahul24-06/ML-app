# ML-app
https://ml-app-heart-disease.streamlit.app/

## A. Problem statement
The objective of this project is to implement and compare multiple machine learning classification models to predict the presence of heart disease using clinical and medical attributes.  

The project involves:
- Training 6 different classification models
- Evaluating them using multiple performance metrics
- Comparing model performance
- Deploying an interactive Streamlit dashboard

## B. Dataset description

Dataset: Heart Disease UCI (Combined Version)  
Instances: 1025  
Features: 13 input features  
Target Variable: `target`

Target meaning:
- 0 → No Heart Disease  
- 1 → Presence of Heart Disease  

### Key Features:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Maximum heart rate
- Exercise induced angina
- ST depression
- Number of vessels
- Thalassemia

The dataset contains both numerical and categorical medical attributes used to predict cardiovascular risk.

## C. Models used: 
The following 6 classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble Boosting)  

All models were trained using an 80-20 train-test split.

---

## Evaluation Metrics Used

Each model was evaluated using:

- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---
### Comparison Table with the evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8033 | 0.8690 | 0.7692 | 0.9091 | 0.8333 | 0.6098 |
| Decision Tree | 0.7049 | 0.6975 | 0.7027 | 0.7879 | 0.7429 | 0.4029 |
| KNN | 0.8033 | 0.8631 | 0.7692 | 0.9091 | 0.8333 | 0.6098 |
| Naive Bayes | 0.8197 | 0.8755 | 0.7895 | 0.9091 | 0.8451 | 0.6410 |
| Random Forest | 0.8361 | 0.9091 | 0.7805 | 0.9697 | 0.8649 | 0.6882 |
| XGBoost | 0.8033 | 0.8561 | 0.7561 | 0.9394 | 0.8378 | 0.6181 |


### Model Performance Observations

#### Logistic Regression
Performed well as a baseline linear classifier. It achieved strong recall (0.9091), meaning it correctly identified most patients with heart disease. Balanced overall performance.

#### Decision Tree
Performed the weakest among all models. Lower accuracy and MCC indicate possible overfitting and instability.

#### KNN
Produced similar results to Logistic Regression. Sensitive to feature scaling but performed well after standardization.

#### Naive Bayes
Delivered strong and stable performance. Achieved good balance between precision and recall due to probabilistic modeling.

#### Random Forest - Best Performing Model
Achieved the highest:
- Accuracy (0.8361)
- AUC (0.9091)
- F1 Score (0.8649)
- MCC (0.6882)

It showed excellent recall (0.9697), meaning it correctly identified almost all heart disease cases. This makes it the most reliable model for this dataset.

#### XGBoost
Performed strongly but slightly below Random Forest. High recall but slightly lower precision.

---
