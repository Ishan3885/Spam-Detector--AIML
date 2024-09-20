Overview
  This project classifies text messages in English, French, and German as either Spam or Ham using machine learning. We leverage Logistic Regression for classification and TF-IDF Vectorization to transform text into numerical features.

Objective
  To build a spam detection model that accurately classifies text data across multiple languages based on provided features.

Dataset
  Text: Messages in multiple languages.
  Labels: "spam" (0) or "ham" (1).
  Data is preprocessed by handling missing values and converting labels to numeric format.

Methodology
  Feature Extraction: TF-IDF Vectorizer to represent text as numerical data.
  Model: Logistic Regression trained on 80% of the dataset.
  Evaluation: Model accuracy and confusion matrix for performance analysis.
  Hyperparameter Tuning: Adjustments made to improve model accuracy.

Results
  Achieved high accuracy in classifying spam and ham messages.
  Confusion matrix analysis to identify misclassifications.

Dependencies
  Python 3.10+
  Pandas, NumPy, Scikit-learn
