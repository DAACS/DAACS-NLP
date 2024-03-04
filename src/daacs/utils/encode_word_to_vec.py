from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.encoders.word_to_vec import WordToVecEncoder
import pandas as pd

b = Bootstrap() 
essays_and_grades  = b.get_essays_and_grades() 

wtv = WordToVecEncoder(bootstrap=b, df=essays_and_grades) 

essays_and_grades_plus_wtv_cols: pd.DataFrame  = wtv \
        .add_tokenized_column()  \
        .add_vectorized_column() \
        .get_data() 

# either of these work. 
print(essays_and_grades_plus_wtv_cols.head(3))


# this returns the original wtv object. so...
# this is also an option...
#
#
# essays_and_grades_plus_wtv_cols: pd.DataFrame  = wtv\
#         .add_tokenized_column()  \
#         .add_vectorized_column() \
#
# then you can do this. 
#
# print(wtv.df.head(2))