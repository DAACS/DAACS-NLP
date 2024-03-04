import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.encoders.word_to_vec import WordToVecEncoder
from daacs.infrastructure.encoders.fast_text import FastTextEncoder

b = Bootstrap() 
eg  = b.get_essays_and_grades(ratings_columns=['TotalScore']) 

print(" *** word2Vec *** ")
wtv = WordToVecEncoder(bootstrap=b, df=eg) 

eg_plus_wtv_cols: pd.DataFrame  = wtv \
        .add_tokenized_column()  \
        .add_vectorized_column() \
        .get_data() 
print(eg_plus_wtv_cols.head(3))


print("*** fastex ***")
ftx = FastTextEncoder(bootstrap=b, df=eg) 
eg_plus_ftx_cols: pd.DataFrame = ftx \
        .add_tokenized_column()  \
        .add_vectorized_column() \
        .get_data() 
print(eg_plus_ftx_cols.head(3))


# this is also an option...
# essays_and_grades_plus_wtv_cols: pd.DataFrame  = wtv\
#         .add_tokenized_column()  \
#         .add_vectorized_column() \
# print(wtv.df.head(2))
