import numpy as np
import tensorflow as tf

def sample_and_shuffle(df, n=2000):
    # Sample 2000 rows for each call_result
    sampled_df = df.groupby('call_result').apply(lambda x: x.sample(min(len(x), n))).reset_index(drop=True)

    # Shuffle the sampled DataFrame
    shuffled_df = sampled_df.sample(frac=1).reset_index(drop=True)

    return shuffled_df

def prepare_dataset(df, tokenizer):
    train_input_texts = df["notes"].tolist()  # Convert to list if it's a Pandas Series

    tokenized_data_train = tokenizer(train_input_texts, return_tensors="np", padding=True)
    # Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
    tokenized_data_train = dict(tokenized_data_train)

    encode_labels = np.array(df["label"].tolist())

    return tokenized_data_train, encode_labels

def to_dataset(tokenized_data, labels, batch):
    dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_data), labels))
    dataset = dataset.batch(batch)  # Adjust batch size as needed

    return dataset