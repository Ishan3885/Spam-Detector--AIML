# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\User\\OneDrive\\Desktop\\CODE\\Projects\\EmailSpamDetector\\Task_1.csv')


df.fillna('', inplace=True)


df['labels'] = df['labels'].astype(str)  
df.loc[df['labels'].str.lower() == 'spam', 'labels'] = 0
df.loc[df['labels'].str.lower() == 'ham', 'labels'] = 1
df['labels'] = df['labels'].astype(int)  


X = df['text']  
Y = df['labels']  


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


vectorizer = TfidfVectorizer(min_df=1, stop_words=['english', 'french', 'german'], lowercase=True)


X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_features, Y_train)


train_predictions = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy}")


test_predictions = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy}")


conf_matrix = confusion_matrix(Y_test, test_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Before Hyperparameter Tuning")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


param_grid = {
    'C': [0.1, 1, 10, 100],  
    'solver': ['liblinear', 'lbfgs']  
}


grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_features, Y_train)


best_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")


tuned_test_predictions = best_model.predict(X_test_features)
tuned_test_accuracy = accuracy_score(Y_test, tuned_test_predictions)
print(f"Tuned Test Accuracy: {tuned_test_accuracy}")


tuned_conf_matrix = confusion_matrix(Y_test, tuned_test_predictions)
sns.heatmap(tuned_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - After Hyperparameter Tuning")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

