# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading dataset
df = pd.read_csv(
    r"C:\Users\ADMIN\OneDrive\Desktop\AI-Internship\AI-INTERN\titanic.csv"
)

# Display first 5 rows
print("\nFIRST 5 ROWS:\n")
print(df.head())

# Dataset information
print("\nDATASET INFORMATION:\n")
print(df.info())

# Missing values
print("\nMISSING VALUES:\n")
print(df.isnull().sum())

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because too many missing values
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# Drop unnecessary columns
drop_columns = ["Name", "Ticket", "PassengerId"]

for col in drop_columns:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Convert text columns into numbers
label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:\n")
print(f"{accuracy * 100:.2f}%")

# Classification report
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(6,4))

sns.countplot(x="Survived", data=df)

plt.title("Titanic Survival Count")

# Show graph
plt.show()

# End of project