from sklearn.feature_extraction.text import TfidfVectorizer
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.string_utils import StringUtils
import nltk 
import pandas as pd

class TfidVectorizerEncoder:
    def __init__(self, df: pd.DataFrame, bootstrap: Bootstrap, output_path:str = None):
        self.output_path = output_path if output_path else bootstrap.ENCODED_DATA_DIR 
        self.bootstrap = bootstrap
        self.df = df  # DataFrame to hold essays

    def tokenize_essay(self, essay, remove_stopwords: bool = False) -> list:
        nltk.download('punkt', quiet=True)  # For tokenization
        tokens = nltk.word_tokenize(essay)
        if remove_stopwords:
            tokens = [word for word in tokens if word.lower() not in nltk.corpus.stopwords.words('english')]
        return tokens

    def add_tokenized_column(self, inbound_text_column='essay', outbound_tokenized_column='tokenized_essay', remove_stopwords: bool = False):
        if self.df is not None and inbound_text_column in self.df.columns:
            self.df[outbound_tokenized_column] = self.df[inbound_text_column].apply(lambda essay: self.tokenize_essay(essay, remove_stopwords))
        else:
            print(f"Data not loaded or column '{inbound_text_column}' not found in DataFrame.")
        return self
    
    def add_vectorized_column(self, inbound_tokenized_column='tokenized_essay', outbound_vectorized_column='jerk'):
        if self.df is not None and inbound_tokenized_column in self.df.columns:
            # Convert tokenized essays to strings
            self.df[inbound_tokenized_column] = self.df[inbound_tokenized_column].apply(lambda x: ' '.join(x))

            # Initialize TfidfVectorizer with custom tokenizer
            tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())

            # Fit and transform the tokenized essays
            vectorized_essays = tfidf_vectorizer.fit_transform(self.df[inbound_tokenized_column])

            # Debug prints
            print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
            print("Tokenized Essays:", self.df[inbound_tokenized_column])

            # Add the vectorized essays to the DataFrame
            self.df[outbound_vectorized_column] = vectorized_essays.toarray().tolist()  # Convert sparse matrix to list
        else:
            print(f"Column '{inbound_tokenized_column}' not found in DataFrame.")
        return self
    
    def get_data(self) -> pd.DataFrame:
        return self.df