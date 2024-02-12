import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
from pyspark.sql.functions import desc, lit, udf, corr, when, lower, col

def main():
    b = Bootstrap() 

    bad_student = """
    As a first-year college student, navigating the transition to university life can be a whirlwind of excitement and newfound freedom. Having recently completed a class on achieving success, I've gained some insights that I know will be useful as I embark on this journey. However, I must admit that my priorities might differ slightly from the conventional definition of success.

    While the class emphasized patterns, discipline, and focus as key elements of success, I find myself drawn to other aspects of the college experience. Sure, recognizing patterns in my academic performance and personal habits is important, but so is the spontaneity and adventure that college life promises. Discipline, too, has its place, but I believe there's value in embracing moments of spontaneity and seizing opportunities as they arise.

    As for focus, well, I suppose I could redirect some of that energy towards my studies. But let's be honest – college isn't just about hitting the books. It's about forging connections, exploring new horizons, and yes, having a bit of fun along the way. And what better way to do that than by socializing, meeting new people, and maybe indulging in the occasional party or two?

    Of course, I understand the importance of balance, and I fully intend to prioritize my academic responsibilities. After all, I wouldn't want to jeopardize my future prospects. But I also believe in living in the moment and making the most of my college experience. And if that means occasionally letting loose and enjoying some of the social aspects of campus life, then so be it.

    In conclusion, while I may not adhere strictly to the traditional notions of success outlined in my recent class, I am confident that I can find my own path to fulfillment in college. By embracing both the academic and social dimensions of the experience, I hope to make the most of these formative years and emerge with memories that will last a lifetime."""

    DAACS_ID="daacs_id"
    METRIC_COLUMN = "TotalScore1"

    ## load the essay data, this should be about the same
    b = Bootstrap()
    spark = b.get_spark() 
    ratings_columns = ['EssayID', 'TotalScore1', 'TotalScore2', 'TotalScore']
    wgu_ratings_raw = spark.read.option("header", True)\
        .csv(b.file_url(WGU_File.wgu_ratings))\
        .select(ratings_columns)\
        .withColumnRenamed("EssayID", DAACS_ID)
    essay_id_counts = wgu_ratings_raw.groupBy(DAACS_ID).count()
    unique_essay_ids = essay_id_counts.filter(col("count") == 1).select(DAACS_ID)
    wgu_ratings = wgu_ratings_raw.join(unique_essay_ids, [DAACS_ID])
    essays_human_rated = spark.read.parquet(b.file_url(WGU_File.essay_human_ratings))\
        .withColumnRenamed("EssayID", DAACS_ID)\
        .join(unique_essay_ids, [DAACS_ID])
    essays_and_grades = essays_human_rated.join(wgu_ratings, [DAACS_ID])\
        .select(METRIC_COLUMN, "essay")\
        .toPandas()    
    data = essays_and_grades #< this is just to minimize. 


    # Set up test and train
    train_df, test_df = train_test_split(data, stratify=data[METRIC_COLUMN], test_size=0.2)
    le = LabelEncoder()
    le.fit(data[METRIC_COLUMN])

    unique_labels = train_df[METRIC_COLUMN].unique()
    num_unique_labels = len(unique_labels)

    train_df['labels'] = le.transform(train_df[METRIC_COLUMN])
    test_df['labels'] = le.transform(test_df[METRIC_COLUMN])

    train_df = train_df[['essay', 'labels']]
    test_df = test_df[['essay', 'labels']]

    model_args = ClassificationArgs(num_train_epochs=3, output_dir=b.BERT_DIR, overwrite_output_dir=True)

    # Use BERT instead of DistilBERT
    if os.path.exists(b.BERT_DIR) and os.listdir(b.BERT_DIR):
        print("Loading the existing model...")
        model = ClassificationModel("bert", b.BERT_DIR, args=model_args, use_cuda=False)
    else:
        print("Training a new model...")
        model = ClassificationModel("bert", "bert-base-uncased", num_labels=num_unique_labels, args=model_args, use_cuda=False)
        model.train_model(train_df)

    print("This is how our student is doing doing?")
    predictions, raw_outputs = model.predict([bad_student])
    predicted_label = le.inverse_transform(predictions)
    print(predicted_label)
        
    print("this is how our model is doing")
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    print(result) 

    print("this is more about how our model is doing, using  sklearn's neat classification report to get more metrics")
    from sklearn.metrics import classification_report
    preds, probs = model.predict(list(test_df['essay'].values))
    print(classification_report(test_df['labels'], preds))


if __name__ == '__main__':
    main()