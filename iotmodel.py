# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data =  "/kaggle/input/cropirrigationscheduling/datasets - datasets.csv"
df = pd.read_csv(data)
df.head()

df['CropType'].value_counts()
df.shape

df.isnull().sum()

df.duplicated().sum()

df.describe()

df

from sklearn.preprocessing import LabelEncoder

categorical_columns = ['CropType']
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()  # Save encoder for each column
    df[column] = label_encoders[column].fit_transform(df[column])  # Encode directly on the column

    # Print mapping of categories to numeric values
    print(f"Mapping for column {column}:")
    for class_, value in zip(label_encoders[column].classes_, range(len(label_encoders[column].classes_))):
        print(f"  {class_} -> {value}")
print("\nData after encoding:")
print(df.head())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df[['SoilMoisture', 'temperature', 'Humidity','CropDays']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Correlations")
plt.show()
df_encoded = pd.get_dummies(df, columns=['CropType'], drop_first=True)
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()

df=df.drop(columns=['CropDays'])

df

# Splitting data
X = df.drop(columns=['Irrigation'])
y = df['Irrigation']

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Decision Tree model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Splitting Data for Naive Bayes
X_bayes = df.drop(columns=['Irrigation'])
y_bayes = df['Irrigation']

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Split data into training and testing sets
X_bayes_train, X_bayes_test, y_bayes_train, y_bayes_test = train_test_split(X_bayes, y_bayes, test_size=0.3, random_state=42)

# Create and train Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_bayes_train, y_bayes_train)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Decision Tree model with limited max_depth
model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Accuracy on training and testing data
train_accuracy = model.score(X_train, y_train)
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Display accuracy results
print(f"Accuracy on Training Data: {train_accuracy:.4f}")
print(f"Accuracy on Testing Data: {test_accuracy:.4f}")


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Decision Tree model with limited max_depth
model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_depth=5,min_samples_split=10, min_samples_leaf=5)

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)


print(f"Accuracy on Testing Data: {test_accuracy:.4f}")

df.duplicated().sum()

df.drop_duplicates()

#Retry
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.38, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')

model.fit(X_train, y_train)



y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)


print(f"Accuracy on Testing Data: {test_accuracy:.4f}")

from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_test,y_pred))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize decision tree
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,   # Feature names
    class_names=['No Irrigation Needed', 'Irrigation Needed'],  # Class names
    filled=True,               # Color based on class
    rounded=True               # Nodes with rounded corners
)
plt.title("Decision Tree Visualization")
plt.show()

from sklearn.model_selection import cross_val_score
model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_depth=5, min_samples_split=10, min_samples_leaf=5)
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print(f"Average cross-validation score: {cv_scores.mean():.4f}")

import pandas as pd

predictions = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

mismatches = (predictions['Actual'] != predictions['Predicted']).sum()

print(predictions.head(10))  # Display the first 10 rows
print(f"\nNumber of mismatched data in the entire dataset: {mismatches}")

# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Irrigation', 'Irrigation'],
            yticklabels=['No Irrigation', 'Irrigation'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


y_bayes_pred = gnb.predict(X_bayes_test)

print("Accuracy:", accuracy_score(y_bayes_test, y_bayes_pred))
print(classification_report(y_bayes_test, y_bayes_pred))

# Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(gnb, X_bayes, y_bayes, cv=5)  # cv=5 means using 5 folds
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

import pandas as pd

# Create DataFrame to compare predictions and actual values
predictions = pd.DataFrame({
    'Actual': y_bayes_test,
    'Predicted': y_bayes_pred
})

# Count the number of mismatched data
mismatches = (predictions['Actual'] != predictions['Predicted']).sum()

# Display prediction table
print(predictions.head(10))  # Display the first 10 rows
print(f"\nNumber of mismatched data in the entire dataset: {mismatches}")

# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create confusion matrix
conf_matrix = confusion_matrix(y_bayes_test, y_bayes_pred)

# Display confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Irrigation', 'Irrigation'],
            yticklabels=['No Irrigation', 'Irrigation'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print("####################")
print("# DATA PREPARATION #")
print("####################")

# Drop CropDays as specified
df = df.drop(columns=['CropDays'])

# Splitting data - use ONE consistent split for all models
X = df.drop(columns=['Irrigation'])
y = df['Irrigation']

# Create ONE train/test split to use for all models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Feature scaling - prepare scaled versions of the train and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform the test set, don't fit on it

print("\n####################")
print("# BASELINE MODELS #")
print("####################")

# Create and train Decision Tree model
model_dt = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', 
                                max_depth=5, min_samples_split=10, min_samples_leaf=5)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(classification_report(y_test, y_pred_dt))

# Create and train Naive Bayes model using the SAME train/test split
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_nb = gnb.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
print(classification_report(y_test, y_pred_nb))

# Create confusion matrices
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

# Fix the confusion matrix visualization error by using explicit labels
def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Irrigation', 'Irrigation'],
               yticklabels=['No Irrigation', 'Irrigation'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

# Plot confusion matrices for baseline models
plot_confusion_matrix(conf_matrix_dt, "Confusion Matrix - Decision Tree")
plot_confusion_matrix(conf_matrix_nb, "Confusion Matrix - Naive Bayes")

print("\n####################")
print("# MODEL ENHANCEMENT #")
print("####################")

print("\n---- Advanced Models Performance ----")

# Random Forest model - using SAME train/test split but scaled features
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)  # Use y_train from the same split
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(classification_report(y_test, rf_pred))

# Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)  # Use y_train from the same split
gb_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
print(classification_report(y_test, gb_pred))

# SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)  # Use y_train from the same split
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(classification_report(y_test, svm_pred))

# 3. Hyperparameter tuning for best model (determine which performed best)
best_model_name = max(['Random Forest', 'Gradient Boosting', 'SVM'], 
                     key=lambda x: [rf_accuracy, gb_accuracy, svm_accuracy][
                         ['Random Forest', 'Gradient Boosting', 'SVM'].index(x)])
print(f"\nBest model: {best_model_name}")

# Set up parameters for grid search based on best model
if best_model_name == 'Random Forest':
    print("\n---- Tuning Random Forest ----")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    grid_model = RandomForestClassifier(random_state=42)
    
elif best_model_name == 'Gradient Boosting':
    print("\n---- Tuning Gradient Boosting ----")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_model = GradientBoostingClassifier(random_state=42)
    
else:  # SVM
    print("\n---- Tuning SVM ----")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    grid_model = SVC(probability=True, random_state=42)

# Perform grid search with cross-validation (this may take some time)
print("Starting grid search (this may take a few minutes)...")
grid_search = GridSearchCV(grid_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)  # Use all data for finding best parameters

# Print best parameters
print("\nBest parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model on test data
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)  # Refit on training data
best_pred = best_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, best_pred)
print(f"\nOptimized {best_model_name} Accuracy: {best_accuracy:.4f}")
print(classification_report(y_test, best_pred))

# 4. Feature Importance Analysis (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Plot feature importances
    importances = best_model.feature_importances_
    sorted_indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance - {best_model_name}')
    plt.bar(range(X.shape[1]), importances[sorted_indices])
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Print feature importances
    print("\nFeature Importances:")
    for i, feature in enumerate([X.columns[i] for i in sorted_indices]):
        print(f"{feature}: {importances[sorted_indices][i]:.4f}")

# 5. Confusion Matrix for best model
conf_matrix_best = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues',
           xticklabels=['No Irrigation', 'Irrigation'],
           yticklabels=['No Irrigation', 'Irrigation'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - Optimized {best_model_name}')
plt.show()

# 6. Compare all models
models = {
    'Decision Tree': model,
    'Naive Bayes': gnb,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'SVM': svm_model,
    f'Optimized {best_model_name}': best_model
}

# Calculate accuracy for each model
accuracies = {}
for name, model_obj in models.items():
    if name.startswith('Optimized'):
        accuracies[name] = best_accuracy
    else:
        # Different data needed for different models
        if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVM']:
            X_test_for_model = X_test_scaled if name in ['Random Forest', 'Gradient Boosting', 'SVM'] else X_test
            y_pred_for_model = model_obj.predict(X_test_for_model)
        else:  # Naive Bayes
            y_pred_for_model = y_bayes_pred
            
        accuracies[name] = accuracy_score(y_test, y_pred_for_model)

# Plot model comparison
plt.figure(figsize=(12, 6))
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0.8, 1.0)  # Adjust y-axis to better show differences
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n---- Model Accuracy Comparison ----")
for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")

# Save the best model for future use
import joblib
joblib.dump(best_model, 'c:\\Users\\nanda\\OneDrive\\Desktop\\IOT\\best_irrigation_model.pkl')
joblib.dump(scaler, 'c:\\Users\\nanda\\OneDrive\\Desktop\\IOT\\feature_scaler.pkl')
print("\nBest model and scaler saved to disk")

# 7. Create a simple prediction function to test with new data
def predict_irrigation(crop_type, soil_moisture, temperature, humidity):
    # Create a sample dataframe with the input data
    sample = pd.DataFrame({
        'CropType': [crop_type],
        'SoilMoisture': [soil_moisture],
        'temperature': [temperature],
        'Humidity': [humidity]
    })
    
    # Transform categorical variables as in the training data
    for col in categorical_columns:
        sample[col] = label_encoders[col].transform(sample[col])
    
    # Scale the features
    sample_scaled = scaler.transform(sample)
    
    # Make prediction
    prediction = best_model.predict(sample_scaled)[0]
    probability = best_model.predict_proba(sample_scaled)[0]
    
    return prediction, probability

# Fix the confusion matrix visualization error by using explicit labels
def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Irrigation', 'Irrigation'],
               yticklabels=['No Irrigation', 'Irrigation'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

# Update the Naive Bayes confusion matrix visualization
plot_confusion_matrix(conf_matrix, title="Confusion Matrix - Naive Bayes")

# Example usage
print("\n---- Example Prediction ----")
crop_type = label_encoders['CropType'].classes_[0]  # Get first crop type from encoder
prediction, probability = predict_irrigation(crop_type, 500, 30, 20)
print(f"Crop: {crop_type}")
print(f"Soil Moisture: 500, Temperature: 30, Humidity: 20")
print(f"Prediction: {'Irrigation needed' if prediction == 1 else 'No irrigation needed'}")
print(f"Probability: {probability[1]:.4f} for irrigation, {probability[0]:.4f} for no irrigation")




