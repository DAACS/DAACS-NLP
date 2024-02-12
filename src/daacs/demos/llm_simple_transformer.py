import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
from daacs.infrastructure.bootstrap import Bootstrap
from  torch.utils.tensorboard import SummaryWriter

def main():
    b = Bootstrap() 
    data = pd.read_csv(f'{b.DATA_DIR}/pol/raw_train_biden.csv')
    print(data.head())

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    writer = SummaryWriter(b.TENSOR_LOGS)

    train_df, test_df = train_test_split(data, stratify=data['Stance'], test_size=0.2)
    le = LabelEncoder()
    le.fit(train_df['Stance'])
    train_df['labels'] = le.transform(train_df['Stance'])
    test_df['labels'] = le.transform(test_df['Stance'])

    train_df = train_df[['Tweet', 'labels']]
    test_df = test_df[['Tweet', 'labels']]

    model_args = ClassificationArgs(num_train_epochs=3, output_dir=b.MODEL_DIR, overwrite_output_dir=True)

    # Check if model is already trained and saved

    if os.path.exists(b.MODEL_DIR) and os.listdir(b.MODEL_DIR):
        print("Loading the existing model...")
        model = ClassificationModel(
            "distilbert",b.MODEL_DIR, args=model_args, use_cuda=False
        )
    else:
        print("Training a new model...")
        model = ClassificationModel(
            "distilbert", "distilbert-base-uncased", args=model_args, use_cuda=False
        )
        # writer.add_scalar("loss", )
        model.train_model(train_df)


    anti_biden_tweet = "Ugh, this was true yesterday and it's also true now: Biden is an idiot"
    predictions, raw_outputs = model.predict([anti_biden_tweet])
    le.inverse_transform(predictions)

    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    result


    # you can also use sklearn's neat classification report to get more metrics
    from sklearn.metrics import classification_report
    preds, probs = model.predict(list(test_df['Tweet'].values))
    # preds = le.inverse_transform(preds)
    print(classification_report(test_df['labels'], preds))


if __name__ == '__main__':
    main()