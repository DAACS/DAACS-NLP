# In this file, we take the essays and create a model that can predict actual grades from new essays
# It should then compare the predictions to the actual scores
# For essays graded by more then one person, it should take into account the combined score

# Option: Do this 100 or 1000 times to make the best model, then use the best model (best for all categories)
# Maybe try it with different versions, and save each version, then compare at the end. . . keep the best
import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
from daacs.infrastructure.essays import Essays
from loguru import logger
from pyspark.sql.functions import desc, lit, udf, corr, when, lower, col


DAACS_ID="daacs_id"
b = Bootstrap()
essays_and_grades = b.get_essays_and_grades() # We want to get all columns 

def analyze_results(category):
    model = LinearRegression()

    # Extract the target variable (e.g., "summary" score) from the dictionary
    target_variable = category  # Change this to the desired score

    # Extract the target scores from train_labels
    y_train = [label[target_variable] for label in train_labels]

    # Train the model with the TF-IDF features and target scores
    model.fit(X_train, y_train)  # Use the numeric target variable y_train

    # Test the rest of the essays with the model
    X_test = vectorizer.transform(test_essays)  # Transform the test essays
    predictions = model.predict(X_test)

    # Assuming you have test_labels in a similar format as train_labels
    # Extract the target scores from test_labels
    y_test = [label[target_variable] for label in test_labels]

    # Calculate mean squared error
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error for {category}: {mse}')


# Load training data
train_essays, train_labels = essays_and_grades 
    # Select the columns we need, so select the essays column and then select the column with the score you want (variable category)
    # Get enough for just the training set

# Load testing data
test_essays, test_labels = essays_and_grades
    # Same as above, but just for the testing set

# START OF ADDED

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
y = essays_pd[category]
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_essays)
X_test = vectorizer.transform(test_essays)

# END OF ADDED


# This list is a list of the columns we want to create models for (in grading)
myList = {"summary", "suggest", "structure", "transition", "focus", "cohesion", "correct", "complex", "conventions"}

for category in myList:
    analyze_results(category)

# Future options
    # Use multiple variables in one model to predict other variables?





