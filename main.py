#importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pandas.core.algorithms import duplicated

#reading data from the dataset
pd.set_option('display.max_columns',None)
df = pd.read_csv("./dataset/Titanic-Dataset.csv")
print(df.head(5))

#Step 2: Data Cleaning
#inspecting for missing values
print(df.isnull().sum())

#Sanity check to identify missing duplicated data
print(df.info())
#Finding percentage of missing values
print(df.isnull().sum()/df.shape[0]*100)

#finding duplicated data
print(df.duplicated().sum())

#inputting missing values with mean
print(df.describe().T)

print(df.describe(include="object"))

#Step 3: Handling Outliers
#Boxplots to identify outliers in the fare and age
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()
print(df.columns)

#correlation with heatmap to interpret the relation of the numerical columns
s = df.select_dtypes(include="number").corr()
sns.heatmap(s)
plt.show()

#Step 4: Data Normalization
# Selecting numerical columns to normalize
numerical_cols = ['Age', 'Fare']

min_max_scaler = MinMaxScaler()
df[numerical_cols] = min_max_scaler.fit_transform(df[numerical_cols])

print(df[numerical_cols].head())

# Creating a family_size column by summing SibSp and Parch
df['family_size'] = df['SibSp'] + df['Parch']

# Extracting title from the Name column
df['title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Display the new columns to confirm the changes
print(df[['SibSp', 'Parch', 'family_size', 'title']].head())

# Step 5: Feature Engineering

# 1. Create family_size by summing SibSp and Parch
df['family_size'] = df['SibSp'] + df['Parch']

# 2. Extract title from the Name column
df['title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print(df[['SibSp', 'Parch', 'family_size', 'title']].head())

# Step 6: Feature Selection

# Convert categorical variables to numerical for feature selection
df_encoded = df.copy()
for col in ['Sex', 'Embarked', 'title']:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Dropping of columns not useful for modeling
df_encoded = df_encoded.drop(['Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

# Checking correlation of numerical features
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.show()

# Feature importance using a RandomForest model
X = df_encoded.drop('Survived', axis=1)  # Assuming 'Survived' is the target column
y = df_encoded['Survived']

model = RandomForestClassifier(random_state=0)
model.fit(X, y)

# Getting feature importance scores
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)

# Plotting feature importance
feature_importance.plot(kind='bar')
plt.title("Feature Importance")
plt.show()


# Step 7: Model Building

# Separate features and target variable
X = df_encoded.drop('Survived', axis=1)
y = df_encoded['Survived']

# Split the data into train/test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1. Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=0)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Evaluate Logistic Regression model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log))

# 2. Random Forest Model
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
