
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from daacs.infrastructure.bootstrap import Bootstrap
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import col, lower, regexp_replace, split

b = Bootstrap()
essay_column = "essay_modified"
model_file = 'random_forest_model.joblib'

essays = b.get_essays_and_grades()\
    .withColumn("essay_lower", lower(col("essay")))\
    .withColumn("essay_no_special", regexp_replace(col("essay_lower"), "[^\\w\\s]", ""))\
    .withColumn("essay_words", split(col("essay_no_special"), "\\s+"))

stop_words_remover = StopWordsRemover(inputCol="essay_words", outputCol=essay_column)
essays = stop_words_remover.transform(essays)
essays_pd = essays.toPandas()
essays_pd[essay_column] = essays_pd[essay_column].apply(lambda x: ' '.join(x))

# Convert score to numeric and handle NaN values
essays_pd['TotalScore1'] = pd.to_numeric(essays_pd['TotalScore1'], errors='coerce')
essays_pd.dropna(subset=['TotalScore1'], inplace=True)

# Split data into training and testing sets
X = essays_pd[essay_column]  # Assuming 'essay_modified' is the column with cleaned essays
y = essays_pd['TotalScore1']
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if a trained model exists
if os.path.exists(model_file):
    # Load the trained model
    pipeline = joblib.load(model_file)
else:
    # Create and fit the pipeline only if the model doesn't exist
    pipeline = make_pipeline(
        TfidfVectorizer(),
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    pipeline.fit(train_texts, train_labels)
    # Save the trained model to disk
    joblib.dump(pipeline, model_file)

# Make predictions on the test data
predictions = pipeline.predict(test_texts)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
rmse = sqrt(mse)
print(f'MSE: {mse}, RMSE: {rmse}')