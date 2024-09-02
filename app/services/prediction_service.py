import pandas as pd
from app.models.model import tokenizer, model
import tensorflow as tf

df = pd.read_csv('./app/data/casted_acs_call_history.csv')

def predict_call_result(input_text: str):

    # Tokenize the input text
    tokenized_data_predict = tokenizer(input_text, return_tensors="np", padding=True, truncation=True)

    # Convert BatchEncoding to a dictionary for Keras
    tokenized_data_predict = dict(tokenized_data_predict)

    # Since it's a single input, no need to batch, just expand dimensions
    predict_dataset = tf.data.Dataset.from_tensors(tokenized_data_predict)

    # Generate predictions
    predictions = model.predict(predict_dataset)

    # Extract predicted label (the class with the highest score)
    predicted_label = tf.argmax(predictions.logits, axis=-1).numpy()[0]

    condition = df['label'] == predicted_label

    predicted_label_str = df.loc[condition, 'call_result']

    # Display the predicted labels as strings
    return predicted_label_str.values[0]
