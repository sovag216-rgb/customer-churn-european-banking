# Customer Segmentation & Churn Pattern Analytics in European Banking

## 1. Introduction

Customer churn is an important challenge for banking institutions because losing existing customers can reduce revenue and increase customer acquisition costs. This project analyzes customer data from a European banking dataset to identify patterns associated with customer churn and develop a machine learning model for churn prediction.

## 2. Project Objective

The main objectives of this project are:

- Analyze customer characteristics and churn patterns.
- Identify factors associated with customer attrition.
- Explore churn across age, gender, geography, balance, and product usage.
- Build a machine learning model to predict customer churn.
- Evaluate the predictive performance of the model.
- Develop an interactive Streamlit dashboard for customer churn analysis.

## 3. Dataset

The dataset contains 10,000 European banking customers with information including:

- Customer ID
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card status
- Active Membership status
- Estimated Salary
- Churn/Exited status

The target variable is Exited, where 1 represents a churned customer and 0 represents a customer who remained with the bank.

## 4. Exploratory Data Analysis

The project examined customer churn using several visualizations, including:

- Customer churn distribution
- Age distribution
- Credit score distribution
- Churn by geography
- Churn by gender
- Churn by age group
- Churn by account balance
- Churn by number of products
- Churn by active membership
- Correlation heatmap

The overall churn rate in the dataset was approximately 20.37%, with 2,037 churned customers and 7,963 active customers.

## 5. Data Preprocessing

The data was prepared for machine learning by:

- Separating the target variable from the explanatory variables.
- Handling categorical variables using encoding.
- Removing inappropriate text fields from the model input.
- Splitting the dataset into training and testing sets.
- Applying feature scaling where required.

## 6. Machine Learning Model

A Random Forest Classifier was developed to predict customer churn.

The model was trained using the processed customer information and evaluated on a held-out test dataset.

## 7. Model Performance

The Random Forest model achieved approximately:

- Accuracy: 86.95%
- ROC-AUC: approximately 0.85

A confusion matrix and ROC curve were used to evaluate classification performance.

## 8. Feature Importance

The Random Forest feature-importance analysis was used to identify which customer characteristics contributed most strongly to churn prediction.

Age was among the most influential variables in the analysis, while other customer and account characteristics also contributed to the prediction.

## 9. Interactive Dashboard

A Streamlit dashboard was developed to provide an interactive interface for:

- Viewing customer churn statistics
- Exploring churn patterns
- Reviewing analytical visualizations
- Entering customer information for churn prediction
- Viewing predicted churn probability and recommended customer actions

## 10. Business Insights

The analysis demonstrates that customer churn is associated with several customer and account characteristics. Understanding these patterns can help banks identify customers who may be at higher risk of leaving and design targeted retention strategies.

Potential retention actions include proactive customer communication, personalized engagement, and investigation of customer dissatisfaction.

## 11. Conclusion

This project demonstrates how exploratory data analysis and machine learning can be combined to analyze customer churn in the banking sector. The Random Forest model achieved approximately 86.95% accuracy and an ROC-AUC of approximately 0.85, showing useful predictive capability.

The project also demonstrates the use of Python, Pandas, data visualization, Scikit-learn, and Streamlit for developing a practical data science solution.

## 12. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- GitHub
