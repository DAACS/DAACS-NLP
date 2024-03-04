
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.string_utils import StringUtils
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import nltk
from itertools import chain

class WordToVecEncoder:
    def __init__(self, df: pd.DataFrame, bootstrap: Bootstrap, output_path:str = None):
        self.output_path = output_path if output_path else bootstrap.ENCODED_DATA_DIR 
        self.bootstrap = bootstrap
        self.df = df  # DataFrame to hold essays

    def tokenize_essay(self, essay, remove_stopwords: bool = False) -> list:
        nltk.download('punkt', quiet=True)  # For tokenization
        tokenized_sentences = [StringUtils.encode_sentence(sentence, remove_stopwords) 
                               for sentence in nltk.sent_tokenize(essay)]
        return tokenized_sentences

    def add_tokenized_column(self, column_name='essay', token_column_name='tokenized_essay', remove_stopwords: bool = False):
        if self.df is not None and column_name in self.df.columns:
            self.df[token_column_name] = self.df[column_name].apply(lambda essay: self.tokenize_essay(essay, remove_stopwords))
        else:
            print(f"Data not loaded or column '{column_name}' not found in DataFrame.")
        return self

    def add_vectorized_column(self, tokenized_column='tokenized_essay', vectorized_column='vectorized_essay', vector_size=100):
        if self.df is not None and tokenized_column in self.df.columns:
            # Flatten the list of tokenized essays to a list of sentences
            tokenized_essays = list(chain.from_iterable(self.df[tokenized_column]))

            # Train the Word2Vec model
            model = Word2Vec(sentences=tokenized_essays, vector_size=vector_size, window=5, min_count=1, workers=4)

            # Function to vectorize a tokenized essay
            def vectorize_essay(tokenized_essay):
                essay_vector = [model.wv[word] for sentence in tokenized_essay for word in sentence if word in model.wv]
                return np.mean(essay_vector, axis=0) if essay_vector else np.zeros(vector_size)

            # Apply the vectorize_essay function
            self.df[vectorized_column] = self.df[tokenized_column].apply(vectorize_essay)
        else:
            print(f"Column '{tokenized_column}' not found in DataFrame.")
        return self
    
    def get_data(self) -> pd.DataFrame :
        return self.df

# class WordToVecEncoder: 
#     """
#     Word2Vec: Developed by Google, Word2Vec provides dense word embeddings and captures semantic relationships between words. It comes in two flavors: Continuous Bag of Words (CBOW) and Skip-Gram.

#     """
#     def __init__(self, bootstrap: Bootstrap, output_path:str = None):
#         self.output_path = output_path if output_path else bootstrap.ENCODED_DATA_DIR 
#         self.bootstrap = bootstrap


#     def tokenize_essay(self, essay, remove_stopwords: bool = False) -> list:
#         """
#         Tokenizes an essay into a list of lists of words.
#         Each inner list represents a sentence, and each sentence is broken down into words.

#         :param essay: A string containing the text of the essay.
#         :param remove_stopwords: Boolean to indicate whether to remove stopwords.
#         :return: A list of lists of tokenized words.
#         """
#         import nltk
#         from nltk.tokenize import sent_tokenize
#         nltk.download('punkt', quiet=True)  # For tokenization
#         tokenized_sentences = [StringUtils.encode_sentence(sentence, remove_stopwords) 
#                                for sentence in sent_tokenize(essay)]
#         return tokenized_sentences
    

#     def add_tokenized_column(self, df: pd.DataFrame, column_name='essay', 
#                              token_column_name = 'tokenized_essay', 
#                              remove_stopwords: bool = False) -> pd.DataFrame:
#         """
#         Adds a new column to the DataFrame with tokenized essays.

#         :param df: pandas DataFrame containing an 'essay' column.
#         :param column_name: Name of the column containing essay text. Default is 'essay'.
#         :param remove_stopwords: Boolean to indicate whether to remove stopwords in tokenization.
#         :return: DataFrame with an additional column 'tokenized_essay'.
#         """
#         if column_name in df.columns:
#             df[token_column_name] = df[column_name].apply(lambda essay: self.tokenize_essay(essay, remove_stopwords))
#         else:
#             print(f"Column '{column_name}' not found in DataFrame.")
#         return df
    
#     def add_vectorized_column(self, df, tokenized_column='tokenized_essay', vector_size=100):
#         from gensim.models import Word2Vec

#         """
#         Adds a new column to the DataFrame with Word2Vec vectors for each essay.

#         :param df: pandas DataFrame containing a column with tokenized essays.
#         :param tokenized_column: Name of the column containing tokenized essay text. Default is 'tokenized_essay'.
#         :param vector_size: The size of the Word2Vec vectors.
#         :return: DataFrame with an additional column 'vectorized_essay'.
#         """
#         # Train the Word2Vec model
#         tokenized_essays = df[tokenized_column].tolist()
#         model = Word2Vec(sentences=tokenized_essays, vector_size=vector_size, window=5, min_count=1, workers=4)

#         # Function to vectorize a tokenized essay
#         def vectorize_essay(tokenized_essay):
#             # Aggregate the vectors of the words in the essay
#             essay_vector = [model.wv[word] for word in tokenized_essay if word in model.wv]
#             if essay_vector:
#                 return np.mean(essay_vector, axis=0)
#             else:
#                 return np.zeros(vector_size)

#         # Apply the vectorize_essay function to each tokenized essay
#         if tokenized_column in df.columns:
#             df['vectorized_essay'] = df[tokenized_column].apply(vectorize_essay)
#         else:
#             print(f"Column '{tokenized_column}' not found in DataFrame.")

#         return df

# # GloVe (Global Vectors for Word Representation): Developed by Stanford, GloVe is similar to Word2Vec but is based on word co-occurrence matrices. It's particularly good at capturing more global word-word relationships.

# # FastText: Created by Facebook, FastText extends Word2Vec to consider subword information (like character n-grams), making it effective for handling out-of-vocabulary words and understanding morphologically rich languages.

# # BERT (Bidirectional Encoder Representations from Transformers): Developed by Google, BERT is a transformer-based model known for capturing contextual information from both directions (left and right context). It's highly effective but also computationally more intensive.

# # ELMo (Embeddings from Language Models): ELMo is a deep contextualized word representation that models both complex characteristics of word use (like syntax and semantics) and how these uses vary across linguistic contexts.

# # Universal Sentence Encoder (USE): Developed by Google, USE is designed to provide strong sentence-level embeddings efficiently. It’s particularly useful if you want to encode full sentences or paragraphs instead of just words.

# # T5 (Text-To-Text Transfer Transformer): T5 converts all NLP tasks into a text-to-text format, making it a versatile choice for various applications.

# # When choosing an encoder, consider the following:

# # Task Complexity: For simpler tasks, models like Word2Vec or GloVe might suffice. For more complex tasks requiring understanding of context, models like BERT or ELMo are more suitable.
# # Computational Resources: Some models (like BERT) require significant computational resources. Ensure your environment can handle these requirements.
# # Data Size: Some models perform better with larger datasets. If you have a smaller dataset, simpler models might be more effective.
# # Fine-Tuning: Consider whether you have the resources and data to fine-tune models like BERT, or if you'd prefer a model that works well out-of-the-box.
# # Each of these models has its strengths and trade-offs, so you might want to experiment with a few to see which works best for your specific dataset and task.