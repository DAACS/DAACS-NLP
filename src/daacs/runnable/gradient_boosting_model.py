import os
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingRegressor  # Import GradientBoostingRegressor for Gradient Boosting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.decorators.timeit import timeit

# Initialize and process data
b = Bootstrap()
nltk.download("stopwords")
essay_column = "essay_modified"
model_file = 'gradient_boosting_model.joblib'

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
        GradientBoostingRegressor(n_estimators=100, random_state=42)
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
print(f'MSE: {mse}, RMSE: {rmse}')
