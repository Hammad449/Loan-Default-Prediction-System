"""
Author: Hammad
When: April 7, 2024
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from carousel import Carousel

# ------------- Part 1: Data Cleaning -----------------
def missing_values(filename):
    data = []

    with open(filename, 'r') as file:
        header = file.readline().strip().split(',')
        for line in file:
            row = line.strip().split(',')
            if len(row) == len(header) and '' not in row:
                data.append(row)

    return data, header

def remove_age_90_plus(data):
    cleaned = []
    count = 0
    for row in data:
        try:
            age = int(row[0])
            if age < 90:
                cleaned.append(row)
            else:
                count += 1
        except ValueError:
            continue
    print(f"Number of records with age > 90: {count}")
    print(f"Remaining number of rows: {len(cleaned)}")
    return cleaned

# ------------- Part 2: Visualization -----------------
def histogram_by_age(data):
    default_ages = []
    not_default_ages = []
    for row in data:
        if row[8] == "1":
            default_ages.append(int(row[0]))
        elif row[8] == "0":
            not_default_ages.append(int(row[0]))

    bins = [20,30,40,50,60,70,80,90]

    plt.hist(default_ages, bins=bins, color='red', edgecolor='black')
    plt.title("Loans in Default")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

    plt.hist(not_default_ages, bins=bins, color='green', edgecolor='black')
    plt.title("Loans Not in Default")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

def plot_homeowner_pie(data):
    def_own = 0
    notdef_own = 0
    for row in data:
        if row[2] == "OWN":
            if row[8] == "1":
                def_own += 1
            elif row[8] == "0":
                notdef_own += 1

    labels = ['Defaulted', 'Not Defaulted']
    sizes = [def_own, notdef_own]
    colors = ['red', 'green']

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title("Homeowners: Default vs Not Defaulted")
    plt.axis('equal')
    plt.show()

def class_distribution(data):
    defaulted = sum(1 for row in data if row[8] == "1")
    not_defaulted = sum(1 for row in data if row[8] == "0")
    print(f"Number of borrowers who defaulted: {defaulted}")
    print(f"Number of borrowers who did not default: {not_defaulted}")

# ------------- Part 3: Model -----------------
def scale(data, scaler=None, fit=True):
    x, y = [], []
    for row in data:
        try:
            loan_amnt = float(row[6])
            income = float(row[1])
            credit_hist = float(row[11])
            x.append([loan_amnt, income, credit_hist])
            y.append(int(row[8]))
        except ValueError:
            continue

    if fit:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)

    return x, y, scaler

def decision_tree(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("\n--- Evaluation ---")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return clf

# ------------- Part 4: Loan Request Predictions + Carousel -----------------
def predict_requests(model, scaler):
    data, header = missing_values("loan_requests.csv")
    valid_rows = []
    x_data = []

    for row in data:
        try:
            loan_amnt = float(row[7])
            income = float(row[2])
            credit_hist = float(row[11])
            x_data.append([loan_amnt, income, credit_hist])
            valid_rows.append(row)
        except ValueError:
            continue

    if not x_data:
        print("No valid rows for prediction.")
        return None

    x_scaled = scaler.transform(x_data)
    predictions = model.predict(x_scaled)

    carousel = Carousel()
    for i in range(len(predictions)):
        loan_dict = {
            'borrower': valid_rows[i][0],
            'person_age': valid_rows[i][1],
            'person_income': valid_rows[i][2],
            'person_home_ownership': valid_rows[i][3],
            'person_emp_length': valid_rows[i][4],
            'loan_intent': valid_rows[i][5],
            'loan_grade': valid_rows[i][6],
            'loan_amnt': valid_rows[i][7],
            'loan_int_rate': valid_rows[i][8],
            'loan_percent_income': valid_rows[i][9],
            'cb_person_default_on_file': valid_rows[i][10],
            'cb_person_cred_hist_length': valid_rows[i][11],
            'prediction': int(predictions[i])
        }
        carousel.add(loan_dict)

    print("Carousel built with all predictions.")
    return carousel

def navigate_carousel(carousel):
    input("\nPress Enter to begin viewing loan predictions...")
    while True:
        data = carousel.getCurrentData()
        if not data:
            print("Empty carousel.")
            return

        print("\n" + "-" * 55)
        print(f"Borrower: {data['borrower']}")
        print(f"Age: {data['person_age']}")
        print(f"Income: ${data['person_income']}")
        print(f"Home_ownership: {data['person_home_ownership']}")
        print(f"Employment: {data['person_emp_length']}")
        print(f"Loan intent: {data['loan_intent']}")
        print(f"Loan grade: {data['loan_grade']}")
        print(f"Amount: ${data['loan_amnt']}")
        print(f"Interest Rate: {data['loan_int_rate']}")
        print(f"Loan percent income: {data['loan_percent_income']}")
        print(f"Historical Defaults: {'Yes' if data['cb_person_default_on_file'] == 'Y' else 'No'}")
        print(f"Credit History: {data['cb_person_cred_hist_length']} years")
        print("-" * 55)
        prediction = int(data['prediction'])
        print(f"Predicted loan_status: Will {'default' if prediction == 1 else 'not default'}")
        print(f"Recommend: {'Reject' if prediction == 1 else 'Accept'}")
        print("-" * 55)

        command = input("Enter 1 (next), 2 (prev), 0 (quit): ")
        if command == '1':
            carousel.moveNext()
        elif command == '2':
            carousel.movePrevious()
        elif command == '0':
            print("Exiting.")
            break
        else:
            print("Invalid input.")

# ------------- Main -----------------
def main():
    train_data, _ = missing_values("credit_risk_train.csv")
    train_data = remove_age_90_plus(train_data)
    histogram_by_age(train_data)
    plot_homeowner_pie(train_data)
    class_distribution(train_data)

    x_train, y_train, scaler = scale(train_data)

    test_data, _ = missing_values("credit_risk_test.csv")
    test_data = remove_age_90_plus(test_data)
    x_test, y_test, _ = scale(test_data, scaler=scaler, fit=False)

    clf = decision_tree(x_train, y_train, x_test, y_test)

    carousel = predict_requests(clf, scaler)
    if carousel:
        navigate_carousel(carousel)

if __name__ == '__main__':
    main()

