import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.encoders.word_to_vec import WordToVecEncoder
from daacs.infrastructure.encoders.fast_text import FastTextEncoder
from daacs.infrastructure.encoders.tfidvectorizer import TfidVectorizerEncoder


print("########## isolated way  ##########")
b = Bootstrap() 

print("*** word2Vec *** ")
eg1  = b.get_essays_and_grades() 
wtv = WordToVecEncoder(bootstrap=b, df=eg1) 
eg_plus_wtv_cols: pd.DataFrame  = wtv \
        .add_tokenized_column()  \
        .add_vectorized_column() \
        .get_data() 
print(eg_plus_wtv_cols.head(3))

print("*** fastex ***")
eg2  = b.get_essays_and_grades() 
ftx = FastTextEncoder(bootstrap=b, df=eg2) 
eg_plus_ftx_cols: pd.DataFrame = ftx \
        .add_tokenized_column()  \
        .add_vectorized_column() \
        .get_data() 
print(eg_plus_ftx_cols.head(3))


print("*** sklearn tfidvectorizer  ***")
eg3  = b.get_essays_and_grades() 
tfid  = TfidVectorizerEncoder(bootstrap=b, df=eg3) 
eg_plus_tfvec_cols = tfid \
        .add_tokenized_column()  \
        .add_vectorized_column() \
        .get_data() 
print(eg_plus_tfvec_cols.head(3))


print("########## single df  way ##########")

eg4  = b.get_essays_and_grades() 

print("*** word2Vec *** ")
wtv = WordToVecEncoder(bootstrap=b, df=eg4) 
wtv \
  .add_tokenized_column(remove_stopwords=True, inbound_text_column='essay', outbound_tokenized_column='tokenized')  \
  .add_vectorized_column(inbound_tokenized_column='tokenized', outbound_vectorized_column='w2v_vectors')

print("*** fastex ***")
ftx = FastTextEncoder(bootstrap=b, df=eg4) 
ftx \
  .add_tokenized_column(remove_stopwords=True, inbound_text_column='essay', outbound_tokenized_column='tokenized')  \
  .add_vectorized_column(inbound_tokenized_column='tokenized', outbound_vectorized_column='fb_vectors') \

print("*** sklearn tfidvectorizer ***")
tfid  = TfidVectorizerEncoder(bootstrap=b, df=eg4) 

all_three = tfid \
  .add_tokenized_column(remove_stopwords=True, inbound_text_column='essay', outbound_tokenized_column='tokenized')  \
  .add_vectorized_column(inbound_tokenized_column='tokenized', outbound_vectorized_column='scikit_vectors') \
  .get_data() 

print(all_three.columns)





