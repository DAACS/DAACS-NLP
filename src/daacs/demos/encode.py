import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.encoders.word_to_vec import WordToVecEncoder
from daacs.infrastructure.encoders.fast_text import FastTextEncoder
from daacs.infrastructure.encoders.tfidvectorizer import TfidVectorizerEncoder


## this bolts columns onto the end of essays and grades.. 
## So they're all there, because there's only one essays and grades, eg. 

b = Bootstrap() 
eg  = b.get_essays_and_grades() 

print(" *** word2Vec *** ")
wtv = WordToVecEncoder(bootstrap=b, df=eg) 

wtv \
        .add_tokenized_column(remove_stopwords=True, inbound_text_column='essay', outbound_tokenized_column='tokenized')  \
        .add_vectorized_column(inbound_tokenized_column='tokenized', outbound_vectorized_column='w2v_vectors')
print("*** fastex ***")

ftx = FastTextEncoder(bootstrap=b, df=eg) 
ftx \
        .add_tokenized_column(remove_stopwords=True, inbound_text_column='essay', outbound_tokenized_column='tokenized')  \
        .add_vectorized_column(inbound_tokenized_column='tokenized', outbound_vectorized_column='fb_vectors') \

print("*** sklearn tfidvectorizer  ***")
tfid  = TfidVectorizerEncoder(bootstrap=b, df=eg) 

all_three = tfid \
        .add_tokenized_column(remove_stopwords=True, inbound_text_column='essay', outbound_tokenized_column='tokenized')  \
        .add_vectorized_column(inbound_tokenized_column='tokenized', outbound_vectorized_column='scikit_vectors') \
        .get_data() 




all_three






