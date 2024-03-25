from gensim.models import FastText
import numpy as np
import pandas as pd
import nltk
from itertools import chain
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.string_utils import StringUtils

class FastTextEncoder:
    def __init__(self, df: pd.DataFrame, bootstrap: Bootstrap, output_path:str = None):
        self.output_path = output_path if output_path else bootstrap.ENCODED_DATA_DIR 
        self.bootstrap = bootstrap
        self.df = df  # DataFrame to hold essays

    def tokenize_essay(self, essay, remove_stopwords: bool = False) -> list:
        nltk.download('punkt', quiet=True)  # For tokenization
        tokenized_sentences = [StringUtils.encode_sentence(sentence, remove_stopwords) 
                               for sentence in nltk.sent_tokenize(essay)]
        return tokenized_sentences

    def add_tokenized_column(self, inbound_text_column='essay', outbound_tokenized_column='tokenized', remove_stopwords: bool = False):
        if self.df is not None and inbound_text_column in self.df.columns:
            self.df[outbound_tokenized_column] = self.df[inbound_text_column].apply(lambda essay: self.tokenize_essay(essay, remove_stopwords))
        else:
            print(f"Data not loaded or column '{inbound_text_column}' not found in DataFrame.")
        return self

    def add_vectorized_column(self, inbound_tokenized_column='tokenized', outbound_vectorized_column='vectorized', vector_size=100):
        if self.df is not None and inbound_tokenized_column in self.df.columns:
            # Flatten the list of tokenized essays to a list of sentences
            tokenized_essays = list(chain.from_iterable(self.df[inbound_tokenized_column]))

            # Train the FastText model
            model = FastText(sentences=tokenized_essays, vector_size=vector_size, window=5, min_count=1, workers=4)

            # Function to vectorize a tokenized essay
            def vectorize_essay(tokenized_essay):
                essay_vector = [model.wv[word] for sentence in tokenized_essay for word in sentence if word in model.wv]
                return np.mean(essay_vector, axis=0) if essay_vector else np.zeros(vector_size)

            # Apply the vectorize_essay function

            self.df[outbound_vectorized_column] = self.df[inbound_tokenized_column].apply(vectorize_essay)
        else:
            print(f"Column '{inbound_tokenized_column}' not found in DataFrame.")
        return self

    def get_data(self) -> pd.DataFrame:
        return self.df