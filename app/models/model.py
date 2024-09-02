import tensorflow as tf
from transformers import BertTokenizer, TFAutoModelForSequenceClassification

# Load the fine-tuned model
model = TFAutoModelForSequenceClassification.from_pretrained('./app/my_finetuned_model')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('./app/my_finetuned_model')
