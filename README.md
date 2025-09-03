# Telco Customer Churn Prediction

## Project Overview
This project predicts telecom customer churn using the Telco Customer Churn dataset.
It includes data preprocessing, EDA, feature engineering, and SMOTEENN for class imbalance.
A Random Forest Classifier was tuned with grid search and cross-validation for optimal performance.
The model achieved a 30% lift in churn F1, improving detection of at-risk customers, and was deployed with Flask.



## Dataset
The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) contains customer information such as demographics, account details, and services used. The target variable is `Churn`, indicating whether a customer has left the company (`Yes`) or not (`No`).

Key features include:
- `customerID`: Unique customer identifier
- `tenure`: Number of months the customer has stayed
- `MonthlyCharges`: Monthly bill amount
- `TotalCharges`: Total amount charged to the customer
- Other categorical features like `Contract`, `PaymentMethod`, `InternetService`, etc.

## Preprocessing
The preprocessing steps include:
1. **Handling Missing Values**: Convert `TotalCharges` to numeric and fill missing values with the median.
2. **Feature Engineering**:
   - Binning `tenure` into 6 quantile-based categories due to its non-normal distribution.
   - Dropping `customerID` as it is not predictive.
3. **Encoding**: One-hot encoding for categorical variables.
4. **Class Imbalance**: Addressed using SMOTEENN (Synthetic Minority Oversampling Technique + Edited Nearest Neighbors).
5. **Data Splitting**: 80-20 train-test split with stratification to maintain churn ratio.

## Exploratory Data Analysis
EDA was performed to understand the data and relationships:
- **Churn Distribution**: Visualized using a bar plot showing the percentage of churned vs. non-churned customers.
- **Feature Correlations**: Bar plot of correlations between features and `Churn`.
- **Monthly and Total Charges**:
  - Scatter plot to show the relationship between `MonthlyCharges` and `TotalCharges`.
  - KDE plots to compare distributions of `MonthlyCharges` and `TotalCharges` for churned vs. non-churned customers.
- **Key Insights**:
  - Higher `MonthlyCharges`, lower `tenure`, and lower `TotalCharges` are associated with higher churn rates.
  - Pair plots and heatmaps were used to visualize feature interactions.

## Model Building
A Random Forest Classifier was chosen due to its robustness with categorical features, imbalanced data, and non-linear relationships. The model pipeline includes:
1. **Baseline Model**: Random Forest with default parameters and class weighting.
2. **SMOTEENN**: Applied to handle class imbalance by oversampling the minority class and cleaning noisy samples.
3. **Hyperparameter Tuning**: GridSearchCV to optimize parameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).
4. **Evaluation**: Models were evaluated using classification reports (precision, recall, F1-score) and confusion matrices.

## Model Deployment
The trained model is deployed using a Flask web application:
- **Input**: User provides customer details via a web form.
- **Preprocessing**: Input data is processed (e.g., binning `tenure`, one-hot encoding).
- **Prediction**: The model predicts whether the customer will churn (`Yes` or `No`).
- **Output**: Prediction is displayed on the web interface.


## File Structure
```
├── WA_Fn-UseC_-Telco-Customer-Churn.csv 
├── app.py                               
├── model.pkl                          
├── columns.pkl                         
├── bin_edges.pkl                        
├── templates/
│   └── index.html                       
├── requirements.txt                     
├── README.md                       
```


## Results
- **Baseline Random Forest**:
  - Precision, recall, and F1-score for churn (`1`) and no-churn (`0`) classes.
  - Confusion matrix to evaluate false positives/negatives.
- **SMOTEENN Random Forest**:
  - Improved performance on the minority class (churn) due to balanced data.
- **Hyperparameter-Tuned Model**:
  - Best parameters from GridSearchCV.
  - Improved F1-score and balanced performance across classes.
- **Cross-Validation**: Mean F1-score on the training set to ensure robustness.

## Dependencies
The project requires the following Python packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
flask
```
## Installation
To run this project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   To run the Flask app:
```bash
python app.py
```
Access the app at `http://127.0.0.1:5000`.


Install them using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn flask
```
