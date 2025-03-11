# SENTIMENT-ANALYSIS-WITH-NLP
"COMPANY" : CODTECH IT SOLUTIONS
"NAME" : SRIMURUGAN 
"INTERN ID" : CT08VRH
"DOMAIN" : MACHINE LEARNING 
"DURATION" : 4 WEEKS 
"MENTOR" : Muzammil Ahmed


This script performs sentiment analysis on a set of product reviews using TF-IDF vectorization and a Logistic Regression model. It follows a structured process that includes data preparation, text preprocessing, model training, and evaluation.

Key Steps in the Code:
Importing Required Libraries:

numpy, pandas: For handling data.
matplotlib.pyplot, seaborn: For data visualization.
sklearn.model_selection.train_test_split: For splitting the dataset into training and testing sets.
sklearn.feature_extraction.text.TfidfVectorizer: For converting text into numerical features.
sklearn.linear_model.LogisticRegression: For training a logistic regression model.
sklearn.metrics: For model evaluation, including accuracy, classification report, and confusion matrix.
Creating a Sample Dataset:

The dataset consists of product reviews and their corresponding sentiments (1 for positive, 0 for negative).
The dataset is stored in a Pandas DataFrame.
Splitting the Data:

The dataset is split into 80% training data and 20% test data using train_test_split().
Text Vectorization using TF-IDF:

TfidfVectorizer is used to convert text reviews into numerical representations.
The model is trained on transformed TF-IDF features.
Training the Logistic Regression Model:

A Logistic Regression classifier is trained on the TF-IDF-transformed training data.
This model learns to classify reviews as positive or negative.
Making Predictions and Evaluating Performance:

Predictions are made on the test dataset.
The accuracy score and classification report (precision, recall, F1-score) are printed.
Visualizing the Confusion Matrix:

A heatmap of the confusion matrix is plotted using seaborn.heatmap().
This helps to see how well the model classifies positive and negative reviews.




OUTPUT : ![Image](https://github.com/user-attachments/assets/e64f12d7-f84a-4469-b015-a71efa4a4209)
