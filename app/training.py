import os
import shutil

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TFAutoModelForSequenceClassification
from keras.optimizers import Adam
from keras import Layer
from utils.functions import sample_and_shuffle, prepare_dataset, to_dataset
import numpy as np

df = pd.read_csv('./app/data/acs_call_history_call_result_notes.csv')

df = df.dropna(subset=['notes', 'call_result'])  # Drop rows with missing values
df['notes'] = df['notes'].astype(str)  # Ensure text data is string
df['call_result'] = df['call_result'].astype(str)  # Ensure labels are integer

number_pattern = r'\b\d{10,13}\b'

# Replace numbers with [phoneNumberToken]
df['notes'] = df['notes'].replace(number_pattern, '[phoneNumberToken]', regex=True)


df = sample_and_shuffle(df,2000)

label_mapping = {label: idx for idx, label in enumerate(df["call_result"].unique())}
df["label"] = df["call_result"].map(label_mapping)

new_df = df

num_labels = len(df["call_result"].unique())
list_of_labels = df["call_result"].unique()


#Load Bert Model and Tokenizer

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#Add New Token
tokenizer.add_tokens(['[phoneNumberToken]'])
model.resize_token_embeddings(len(tokenizer))

train_length = round(0.7*len(df))

train_df = df[:train_length]
val_df = df[train_length:]


tokenized_data_train, train_labels = prepare_dataset(train_df,tokenizer)
tokenized_data_val, val_labels = prepare_dataset(val_df,tokenizer)

# Prepare TensorFlow dataset
dataset = to_dataset(tokenized_data_train, train_labels, 20)
validation_dataset = to_dataset(tokenized_data_val, val_labels, 20)


# Define optimizer and loss function
optimizer = Adam(learning_rate=5e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define metrics
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Custom training loop
epochs = 3  # Adjust as needed
for epoch in range(epochs):
    print(f"Start of epoch {epoch+1}")

    # Training loop
    for step, batch in enumerate(dataset):
        with tf.GradientTape() as tape:
            inputs, labels = batch
            logits = model(inputs, training=True).logits
            loss_value = loss_fn(labels, logits)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update and print training accuracy for each step
        train_acc_metric.update_state(labels, logits)
        step_acc = train_acc_metric.result().numpy()
        print(f"Step {step+1} - Training Accuracy: {step_acc:.4f}")

    # Print training accuracy at the end of the epoch
    train_acc = train_acc_metric.result().numpy()
    print(f"Epoch {epoch+1} - Training Accuracy: {train_acc:.4f}")

    # Validation loop
    for val_step, val_batch in enumerate(validation_dataset):
        val_inputs, val_labels = val_batch
        val_logits = model(val_inputs, training=False).logits

        # Update and print validation accuracy
        val_acc_metric.update_state(val_labels, val_logits)

    # Print validation accuracy at the end of the epoch
    val_acc = val_acc_metric.result().numpy()
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}")

    if (train_acc > 0.9 and val_acc > 0.85):
      break

# Directory path
directory = "./app/my_finetuned_model"

# Check if the directory exists
if os.path.exists(directory) and os.path.isdir(directory):
    # Remove the directory if it exists
    shutil.rmtree(directory)
    print(f"The directory {directory} has been removed.")
    os.remove('./app/data/casted_acs_call_history.csv')

# Create the directory
os.makedirs(directory)
print(f"The directory {directory} has been created.")

new_df.to_csv('./app/data/casted_acs_call_history.csv', index=False)

# Save the model
model.save_pretrained('./app/my_finetuned_model')

# Save the tokenizer as well (if you made any changes to it)
tokenizer.save_pretrained('./app/my_finetuned_model')



