import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from daacs.infrastructure.sample_data import SampleData
from daacs.infrastructure.string_utils import StringUtils

# Load the trained model
pipeline = joblib.load('random_forest_model.joblib')

bad_student = StringUtils.clean_sentence(SampleData.bad_student)
new_essay_transformed = pipeline.named_steps['tfidfvectorizer'].transform([bad_student])
prediction = pipeline.named_steps['randomforestregressor'].predict(new_essay_transformed)
print(f"Predicted Score for bad student: {prediction[0]}")
# before cleaning Predicted Score: 19.44 After cleaning  19.51

good_student = StringUtils.clean_sentence(SampleData.good_student)
new_essay_transformed = pipeline.named_steps['tfidfvectorizer'].transform([good_student])
prediction = pipeline.named_steps['randomforestregressor'].predict(new_essay_transformed)
print(f"Predicted Score for good student: {prediction[0]}")