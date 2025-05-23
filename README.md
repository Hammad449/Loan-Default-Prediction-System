# Loan Default Prediction & Visualization Tool

This project analyzes credit risk data, trains a Decision Tree classifier to predict loan defaults, and provides an interactive terminal interface to explore loan approval predictions using a custom-built circular doubly linked list ("carousel").

## Files

- `assignment1.py`: Main driver file for the entire project (data processing, visualization, modeling, prediction).
- `carousel.py`: Implements a circular doubly linked list to navigate loan predictions interactively.
- `credit_risk_train.csv`: Training dataset.
- `credit_risk_test.csv`: Testing dataset.
- `loan_requests.csv`: New loan requests to predict using the trained model.

## Features

- Cleans and preprocesses raw loan data
- Visualizes borrower trends using:
  - Histograms of loan defaults by age
  - Pie charts of homeownership status
- Trains a Decision Tree model using key financial features
- Evaluates model accuracy and prints a full classification report
- Predicts outcomes for new loan applications
- Uses a circular doubly linked list to navigate prediction results in a user-friendly way

## How It Works

1. **Data Cleaning**  
   Removes rows with missing values and filters out borrowers aged 90+ for better model training.

2. **Visualization**  
   Creates histograms and pie charts to explore default trends by age and homeownership.

3. **Model Training**  
   Uses `StandardScaler` for feature scaling and `DecisionTreeClassifier` for prediction.

4. **Prediction and Output**  
   Predicts default likelihood for incoming loan requests and presents each case interactively in a carousel.

## Requirements

Install the following Python packages:

```bash
pip install matplotlib scikit-learn
