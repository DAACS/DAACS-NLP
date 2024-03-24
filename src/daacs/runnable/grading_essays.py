# The purpose of this file is to create models that will evaluate essays
# The models are trained and tested on labeled data
# The best model will be used going forward

# Load libraries
import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
from daacs.infrastructure.essays import Essays
from loguru import logger
#from pyspark.sql.functions import desc, lit, udf, corr, when, lower, col

# Pull in the data, text preprocessing?
DAACS_ID="daacs_id"
b = Bootstrap()
essays_and_grades = b.get_essays_and_grades()



# Create function to analyze results based on the type of result
# Each time you run the code, you can analyze all variables or just some
def analyze_results(category):
    model = LinearRegression()

    # Extract the target variable (e.g., "summary" score) from the dictionary
    target_variable = category  # Change this to the desired score

    # Load training and testing data
    X = essays_and_grades["essay"]
    y = essays_and_grades[category]
    print(X.head())
    print(y.head())

    # Replace NA values
    match category:
        case "Content_Summary":
            df2 = essays_and_grades["Content_Summary1"]
            df3 = essays_and_grades["Content_Summary2"]
        case "Content_Suggestions":
            df2 = essays_and_grades["Content_Suggestions1"]
            df3 = essays_and_grades["Content_Suggestions2"]
        case "Organization_Structure":
            df2 = essays_and_grades["Organization_Structure1"]
            df3 = essays_and_grades["Organization_Structure2"]
        case "Organization_Transition":
            df2 = essays_and_grades["Organization_Transition1"]
            df3 = essays_and_grades["Organization_Transition2"]
        case "Paragraphs_Focus_on_a_Main_Idea":
            df2 = essays_and_grades["Paragraphs_Focus_on_a_Main_Idea1"]
            df3 = essays_and_grades["Paragraphs_Focus_on_a_Main_Idea2"]
        case "Paragraphs_Cohesion":
            df2 = essays_and_grades["Paragraphs_Cohesion1"]
            df3 = essays_and_grades["Paragraphs_Cohesion2"]
        case "Sentences_Correct":
            df2 = essays_and_grades["Sentences_Correct1"]
            df3 = essays_and_grades["Sentences_Correct2"]
        case "Sentences_Complex":
            df2 = essays_and_grades["Sentences_Complex1"]
            df3 = essays_and_grades["Sentences_Complex2"]
        case "Conventions":
            df2 = essays_and_grades["Conventions1"]
            df3 = essays_and_grades["Conventions2"]
        case "TotalScore":
            df2 = essays_and_grades["TotalScore1"]
            df3 = essays_and_grades["TotalScore2"]
        case _:
            print(f"Something is wrong with you category: {category}.")
    filled_y = y.fillna(df2.combine_first(df3))
    #filled_y.columns = y.columns

    # Split train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, filled_y, test_size=0.25, random_state=42)

    # Extract the target scores from train_labels
    print(train_labels.head())
    #y_train = [label[target_variable] for label in train_labels]
    y_train = train_labels.tolist()

    ####y_train = train_labels[target_variable].tolist()


    # Test the rest of the essays with the model
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Assuming you have test_labels in a similar format as train_labels
    # Extract the target scores from test_labels
    ###y_test = [label[target_variable] for label in test_labels]
    y_test = test_labels.tolist()


    # Calculate mean squared error
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error for {category}: {mse}')
    return mse

# List of columns we want to use for modeling
columns_available = ['Content_Summary', 'Content_Suggestions', 'Organization_Structure',
                     'Organization_Transition', 'Paragraphs_Focus_on_a_Main_Idea',
                     'Paragraphs_Cohesion', 'Sentences_Correct', 'Sentences_Complex',
                     'Conventions', 'TotalScore']

# Calling the function above
mse_df = []

for category in columns_available:
    newMSE = analyze_results(category)
    print(f"{newMSE} is my newMSE.")
    mse_df.append({'Category': category, 'MSE': newMSE})

mse_df = pd.DataFrame(mse_df)
print(mse_df)
