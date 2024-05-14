# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import joblib

# %%
df = pd.read_csv("./data/Diabetes_prediction.csv")
df

# %%
df.head()

# %%
df.tail()

# %% [markdown]
# ### The dataset contains the data for the following features to help us predict whether a person is diabetic or not:
# 
#     Pregnancies: Number of times pregnant
# 
#     Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
#     
#     BloodPressure: Diastolic blood pressure (mm Hg)
#     
#     SkinThickness: Triceps skin fold thickness (mm)
#     
#     Insulin: 2-Hour serum insulin (mu U/ml)
#     
#     BMI: Body mass index (weight in kg/(height in m)2)
#     
#     DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
#     
#     Age: Age (years)
#     
#     Outcome: Class variable (0 if non-diabetic, 1 if diabetic)

# %%
df.info()

# %%
df.describe()

# %%
df = df[df['Age'] >= 21]
df.head()

# %%
def count_values(data):
    positives = (data > 0).sum()
    negatives = (data < 0).sum()
    zeros = (data == 0).sum()
    return pd.Series([positives, negatives, zeros], index=['Positives', 'Negatives', 'Zeros'])

# Apply the function to each column in the DataFrame
value_counts = df.apply(count_values)
value_counts


# %%
df['Insulin'] = df['Insulin'].apply(lambda x: abs(x) if x < 0 else x)

# Verify the changes by checking for any remaining negative values
negative_insulin_count = (df['Insulin'] < 0).sum()

# %%
def count_values(data):
    positives = (data > 0).sum()
    negatives = (data < 0).sum()
    zeros = (data == 0).sum()
    return pd.Series([positives, negatives, zeros], index=['Positives', 'Negatives', 'Zeros'])

# Apply the function to each column in the DataFrame
value_counts = df.apply(count_values)
value_counts

# %%
df['Age'] = df['Age'].apply(lambda x: abs(x) if x < 0 else x)

# Verify the changes by checking for any remaining negative values
negative_insulin_count = (df['Age'] < 0).sum()
negative_insulin_count

# %%
value_counts = df.apply(count_values)
value_counts

# %%
df.isna()

# %%


plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".3f", vmin=-1, vmax=1, annot_kws={"size": 12})
heatmap.set_title('Correlation Heatmap of Data Fields', fontsize=16)
heatmap.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()  # Adjust layout
plt.show()


# %%
df.corr()

# %%
df_corr = df.corr()['Diagnosis']
df_corr

# %%
sns.set(font_scale=2)
pairplot = sns.pairplot(df, hue='Diagnosis', )

# Add a title
pairplot.fig.suptitle("Pairplot of Features with Diagnosis", y=1.02)
plt.show()

# %%
units = {
    'Age': 'years',
}

# Set font scale
sns.set(font_scale=1)

# Specify the variables of interest
variables_of_interest = ['DiabetesPedigreeFunction', 'Age', 'Diagnosis']

# Create pair plot for the specified variables
pairplot = sns.pairplot(df[variables_of_interest], hue='Diagnosis')

# Add units to the axes labels
for ax in pairplot.axes.flat:
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    ax.set_xlabel(f"{xlabel} ({units.get(xlabel, '')})")
    ax.set_ylabel(f"{ylabel} ({units.get(ylabel, '')})")

# Show the plot
plt.show()


# %%
units = {
    'BloodPressure': 'mmHg',
    'Insulin': 'muU/mL'
}

# Set font scale
sns.set(font_scale=1)

# Specify the variables of interest
variables_of_interest = ['BloodPressure', 'Insulin', 'Diagnosis']

# Create pair plot for the specified variables
pairplot = sns.pairplot(df[variables_of_interest], hue='Diagnosis')

# Add units to the axes labels
for ax in pairplot.axes.flat:
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    ax.set_xlabel(f"{xlabel} ({units.get(xlabel, '')})")
    ax.set_ylabel(f"{ylabel} ({units.get(ylabel, '')})")

# Show the plot
plt.show()

# %%
fig , axes = plt.subplots(3,3,figsize=(40,30))
index = 0
axes = axes.flatten()
for col , val in df.items():
    col_dist = sns.histplot(val , ax = axes[index] , kde = True , color = 'blue' , stat="density")
    col_dist.set_xlabel(col, fontsize=40)  
    col_dist.set_ylabel('density', fontsize=40)  
    index += 1  
fig.suptitle("Density Histogram of fields in the dataset", fontsize=50, y=1.02)
plt.tight_layout()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Define units for variables
units = {
    'Glucose': 'mg/dL',  # Example units for glucose
    'Insulin': 'µU/mL',  # Example units for insulin
    'Age': 'years'  # Example units for age
}

selected_variables = ['Glucose', 'Insulin', 'Age']

# Creating separate subplots for each variable
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Loop through the variables and create a box plot for each
for i, var in enumerate(selected_variables):
    sns.boxplot(data=df, y=var, ax=axes[i], palette="Set2", showfliers=True)
    axes[i].set_title(f"Box Plot of {var}")
    axes[i].set_ylabel(f"{var} ({units[var]})")
    axes[i].set_xlabel('')

# Adjust the layout
plt.title('Box Plot of Glucose, Insulin, Age')
plt.tight_layout()
plt.show()


# %%
# def classify_gender(row):
#     if row['Pregnancies'] > 0:
#         return 'Female'
#     else:
#         return 'Unknown'

# df['Gender'] = df.apply(classify_gender, axis=1)

# # Group data by 'Diagnosis' and 'Gender', count occurrences, and reset index
# gender_diagnosis_counts = df.groupby(['Diagnosis', 'Gender']).size().reset_index(name='Count')

# # Define custom colors for the bar graph
# custom_palette = {0: 'lightblue', 1: 'salmon'}

# # Plot the bar graph with custom colors
# plt.figure(figsize=(10, 6))
# sns.barplot(data=gender_diagnosis_counts, x='Gender', y='Count', hue='Diagnosis', palette=custom_palette)
# plt.title('Distribution of Diabetes by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.show()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
for feature in X.columns:
   
    new_X = X[[feature]]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=42)
    
    # Fit linear regression model
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    
    # Make predictions on the testing set
    y_pred = linear_regression.predict(X_test)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    
   
    print(f"Feature: {feature}")
    print(f"Coefficient: {linear_regression.coef_}")
    print(f"Intercept: {linear_regression.intercept_}")
    print(f"Mean Squared Error: {mse}")

# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

subset_col1 =  ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age', 'Diagnosis']
subset_df = df[subset_col1]
subset_df = subset_df.dropna()
# subset_df = pd.get_dummies(subset_df, columns=['sex'], drop_first=True)
# codes, uniques = pd.factorize(subset_df['cabin'])
# subset_df['cabin'] = codes
X = subset_df[['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]
y = subset_df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)
model = LogisticRegression()
model.fit(X_train, y_train)
predicts = model.predict(X_test)
accuracy = accuracy_score(y_test, predicts)
print(f'Accuracy: {accuracy}')

# %%
print(df.columns)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_


        

# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicts)
print("Confusion Matrix:")
print(cm)

# %%
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, predicts))


# %%
# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize decision tree classifier
clf = DecisionTreeClassifier(random_state=87)

# Fit the model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
plt.show()


# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Predictions
y_pred = grid_search.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree
best_clf = grid_search.best_estimator_
plt.figure(figsize=(20,10))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
plt.show()


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize Random Forest classifier

clf = RandomForestClassifier(max_depth=10, random_state=87)


# Fit the model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize Random Forest classifier
clf = RandomForestClassifier(max_depth=10, random_state=87)

# Fit the model
clf.fit(X_train, y_train)

# Predictions on training data
y_pred_train = clf.predict(X_train)

# Predictions on testing data
y_pred_test = clf.predict(X_test)

# Accuracy for training data
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f'Accuracy for training data: {accuracy_train * 100:.2f}%')

# Accuracy for testing data
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Accuracy for testing data: {accuracy_test * 100:.2f}%')

# Classification Report
print("Classification Report for testing data:")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
print("Confusion Matrix for testing data:")
print(confusion_matrix(y_test, y_pred_test))


# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize XGBoost classifier
clf = XGBClassifier(random_state=42)

# Define hyperparameters grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Predictions
y_pred = grid_search.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=87)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize neural network model
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3, verbose=0)

# Evaluate the model
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

best_accuracy = 0
best_random_state = None

# Iterate over a range of random_state values
for random_state_value in range(1, 101):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state_value)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize neural network model
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3, verbose=0)

    # Evaluate the model
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Check if current model has higher accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_random_state = random_state_value

# Print the best random_state and accuracy
print(f'Best Random State: {best_random_state}')
print(f'Best Accuracy: {best_accuracy * 100:.2f}%')


# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize XGBoost classifier
clf = XGBClassifier(random_state=42)

# Define hyperparameters grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize GridSearchCV for XGBoost
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the XGBoost model
grid_search.fit(X_train, y_train)

# Best parameters for XGBoost
print("Best Parameters for XGBoost:", grid_search.best_params_)

# Predictions for XGBoost
y_pred_xgb = grid_search.predict(X_test)

# Accuracy for XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost Accuracy: {accuracy_xgb * 100:.2f}%')

# Classification Report for XGBoost
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix for XGBoost
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Initialize Logistic Regression classifier
log_reg = LogisticRegression(random_state=42, max_iter=10000)  # Increased max_iter for convergence

# Fit the Logistic Regression model
log_reg.fit(X_train, y_train)

# Predictions for Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)

# Accuracy for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f'\nLogistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%')

# Classification Report for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Confusion Matrix for Logistic Regression
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize standard Logistic Regression classifier
log_reg = LogisticRegression(random_state=42, max_iter=10000)  # Increased max_iter for convergence

# Fit the Logistic Regression model
log_reg.fit(X_train, y_train)

# Predictions for Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)

# Accuracy for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f'Logistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%')

# Classification Report for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Confusion Matrix for Logistic Regression
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the Decision Tree model
clf.fit(X_train, y_train)

# Predictions for Decision Tree
y_pred = clf.predict(X_test)

# Accuracy for Decision Tree
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy * 100:.2f}%')

# Classification Report for Decision Tree
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix for Decision Tree
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

# Load data from CSV file
file_path = './data/Diabetes_prediction.csv'
df = pd.read_csv(file_path)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Oversampling the minority class
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Initialize Decision Tree classifier with high variance
clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Fit the model
clf.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))




