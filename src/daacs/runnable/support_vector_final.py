import os
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.svm import SVR  # Import SVR for Support Vector Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.decorators.timeit import timeit
from daacs.infrastructure.string_utils import StringUtils


# Initialize and process data
b = Bootstrap()
nltk.download("stopwords")
essay_column = "essay_modified"
model_file = 'svm_model.joblib'  # Change the model file name for SVM

essays_pd = b.get_essays_and_grades()
## Text preprocessing
essays_pd[essay_column] = essays_pd['essay'].apply(StringUtils.clean_sentence)
essays_pd.dropna(subset=['TotalScore1'], inplace=True)

# Prepare training and test data
X = essays_pd[essay_column]
y = essays_pd['TotalScore1']
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

@timeit
def train_model(train_texts, train_labels):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SVR(kernel='linear')  # You can adjust the kernel and other parameters as needed
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline

pipeline = train_model(train_texts, train_labels)

# Save the trained model to disk (this will overwrite the existing file)
joblib.dump(pipeline, model_file)

# Optional: Load and predict for evaluation
pipeline = joblib.load(model_file)
predictions = pipeline.predict(test_texts)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
rmse = sqrt(mse)
print(f'MSE: {mse}, RMSE: {rmse}')


### Testing other kernels with SVM:

# Initialize and process data

essays_pd = b.get_essays_and_grades()
stop_words = set(stopwords.words('english'))

# Text preprocessing
essays_pd['essay_modified'] = essays_pd['essay'].str.lower().str.replace("[^\\w\\s]", "", regex=True)
essays_pd['essay_words'] = essays_pd['essay_modified'].str.split()
essays_pd['essay_modified'] = essays_pd['essay_words'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))
essays_pd['TotalScore1'] = pd.to_numeric(essays_pd['TotalScore1'], errors='coerce')
essays_pd.dropna(subset=['TotalScore1'], inplace=True)

# Prepare training and test data
X = essays_pd[essay_column]
y = essays_pd['TotalScore1']
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

@timeit
def train_model(train_texts, train_labels):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SVR(kernel='poly')  # You can adjust the kernel and other parameters as needed
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline

pipeline = train_model(train_texts, train_labels)

# Save the trained model to disk (this will overwrite the existing file)
joblib.dump(pipeline, model_file)

# Optional: Load and predict for evaluation
pipeline = joblib.load(model_file)
predictions = pipeline.predict(test_texts)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
rmse = sqrt(mse)
print(f'MSE: {mse}, RMSE: {rmse}')



# Initialize and process data
essays_pd = b.get_essays_and_grades()
stop_words = set(stopwords.words('english'))

# Text preprocessing
essays_pd['essay_modified'] = essays_pd['essay'].str.lower().str.replace("[^\\w\\s]", "", regex=True)
essays_pd['essay_words'] = essays_pd['essay_modified'].str.split()
essays_pd['essay_modified'] = essays_pd['essay_words'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))
essays_pd['TotalScore1'] = pd.to_numeric(essays_pd['TotalScore1'], errors='coerce')
essays_pd.dropna(subset=['TotalScore1'], inplace=True)

# Prepare training and test data
X = essays_pd[essay_column]
y = essays_pd['TotalScore1']
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

@timeit
def train_model(train_texts, train_labels):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SVR(kernel='rbf')  # You can adjust the kernel and other parameters as needed
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline

pipeline = train_model(train_texts, train_labels)

# Save the trained model to disk (this will overwrite the existing file)
joblib.dump(pipeline, model_file)

# Optional: Load and predict for evaluation
pipeline = joblib.load(model_file)
predictions = pipeline.predict(test_texts)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
rmse = sqrt(mse)
print(f'MSE: {mse}, RMSE: {rmse}')


# Initialize and process data
essays_pd = b.get_essays_and_grades()
stop_words = set(stopwords.words('english'))

# Text preprocessing
essays_pd['essay_modified'] = essays_pd['essay'].str.lower().str.replace("[^\\w\\s]", "", regex=True)
essays_pd['essay_words'] = essays_pd['essay_modified'].str.split()
essays_pd['essay_modified'] = essays_pd['essay_words'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))
essays_pd['TotalScore1'] = pd.to_numeric(essays_pd['TotalScore1'], errors='coerce')
essays_pd.dropna(subset=['TotalScore1'], inplace=True)

# Prepare training and test data
X = essays_pd[essay_column]
y = essays_pd['TotalScore1']
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

@timeit
def train_model(train_texts, train_labels):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SVR(kernel='sigmoid')  # You can adjust the kernel and other parameters as needed
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline

pipeline = train_model(train_texts, train_labels)

# Save the trained model to disk (this will overwrite the existing file)
joblib.dump(pipeline, model_file)

# Optional: Load and predict for evaluation
pipeline = joblib.load(model_file)
predictions = pipeline.predict(test_texts)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
rmse = sqrt(mse)
print(f'MSE: {mse}, RMSE: {rmse}')


## Here, I add a tokenizer:


# Define a custom tokenizer function
def custom_tokenizer(text):
    # You can modify this tokenizer based on your specific requirements
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
    return words

# Initialize and process data
essays_pd = b.get_essays_and_grades()
stop_words = set(stopwords.words('english'))

# Text preprocessing
essays_pd['essay_modified'] = essays_pd['essay'].str.lower().str.replace("[^\\w\\s]", "", regex=True)
essays_pd['essay_words'] = essays_pd['essay_modified'].str.split()
essays_pd['essay_modified'] = essays_pd['essay_words'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))
essays_pd['TotalScore1'] = pd.to_numeric(essays_pd['TotalScore1'], errors='coerce')
essays_pd.dropna(subset=['TotalScore1'], inplace=True)

# Prepare training and test data
X = essays_pd[essay_column]
y = essays_pd['TotalScore1']
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

@timeit
def train_model(train_texts, train_labels):
    pipeline = make_pipeline(
        TfidfVectorizer(tokenizer=custom_tokenizer),  # Use the custom tokenizer here
        SVR(kernel='linear')
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline

pipeline = train_model(train_texts, train_labels)

# Save the trained model to disk
joblib.dump(pipeline, model_file)

# Optional: Load and predict for evaluation
pipeline = joblib.load(model_file)
predictions = pipeline.predict(test_texts)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
rmse = sqrt(mse)
print(f'Tokenizer evaluation...')
print(f'MSE: {mse}, RMSE: {rmse}')
