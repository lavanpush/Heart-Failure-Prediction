import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import shap

file_path = '/Users/lavanyapushparaj/Desktop/Heart Failure Prediction/heart_failure_clinical_records_dataset.csv'
DF = pd.read_csv(file_path)

# Dropping NaN values
df = DF.dropna()

print(df)

# Correlation matrix
correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Removing outliers
def remove_outliers(df, feature_name, method='z-score', threshold=3):
    if method == 'z-score':
        z_scores = ((df[feature_name] - df[feature_name].mean()) / df[feature_name].std()).abs()
        outliers = df[z_scores > threshold]
    elif method == 'iqr':
        Q1 = df[feature_name].quantile(0.25)
        Q3 = df[feature_name].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[feature_name] < (Q1 - 1.5 * IQR)) | (df[feature_name] > (Q3 + 1.5 * IQR))]
    else:
        raise ValueError("Invalid method. Use 'z-score' or 'iqr'.")

    df_no_outliers = df.drop(outliers.index)
    return df_no_outliers

for column in df.columns:
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot for {column}')
    plt.show()
    df = remove_outliers(df, column, method='z-score', threshold=3)
    df = remove_outliers(df, column, method='iqr')

for column in df.columns:
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot for {column} (After Removing Outliers)')
    plt.show()


# Feature Engineering
bins = [29, 49, 59, 69, 100]
labels = ['30-49', '50-59', '60-69', '70+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# One-hot encoding
df = pd.get_dummies(df, columns=['age_group'], drop_first=True)


# Assigning X and y
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

#Plotting Scatterplot
for feature in X.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature, y='DEATH_EVENT', hue='DEATH_EVENT', palette='viridis')
    plt.title(f'Scatter Plot of {feature} vs. DEATH_EVENT')
    plt.xlabel(feature)
    plt.ylabel('DEATH_EVENT')
    plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize Logistic Regression model
logreg_model = LogisticRegression(random_state=42, max_iter=1000)


# Cross-Validation
cv_scores = cross_val_score(logreg_model, X_train, y_train, cv=15, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy: {:.2f}".format(np.mean(cv_scores)))


# Train & Predict
logreg_model.fit(X_train, y_train)
y_pred = logreg_model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')



# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#shap
background_samples = shap.sample(X_train, 10)
explainer = shap.KernelExplainer(logreg_model.predict_proba, background_samples)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")





