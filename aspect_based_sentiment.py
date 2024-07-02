#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())


remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)


for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])


print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)


raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))


print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1, activation='sigmoid')])

model.summary()


model.compile(loss=losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Attention, Concatenate, Lambda

# Parameters
vocab_size = 10000
maxlen = 250
embedding_dim = 16

# Function to generate synthetic aspect terms for datasets
def generate_aspect_terms(dataset, aspect_term='aspect term'):
    aspect_sequences = []
    for text_batch, _ in dataset:
        aspect_sequences.extend([[aspect_term] * maxlen for _ in range(len(text_batch))])
    return aspect_sequences

# Initialize tokenizer for text and aspect terms
tokenizer = Tokenizer(num_words=vocab_size)

texts = [text.numpy().decode('utf-8') for text_batch, _ in raw_train_ds for text in text_batch]
aspects = generate_aspect_terms(raw_train_ds)

# Fit tokenizer on both text and aspects
tokenizer.fit_on_texts(texts + aspects)

# Tokenize and pad sequences
def preprocess_dataset(dataset, aspect_sequences):
    text_sequences = []
    labels = []
    for text_batch, label_batch in dataset:
        text_seqs = tokenizer.texts_to_sequences([text.numpy().decode('utf-8') for text in text_batch])
        padded_texts = pad_sequences(text_seqs, maxlen=maxlen, padding='post')
        text_sequences.append(padded_texts)
        labels.extend(label_batch.numpy())
    aspect_seqs = tokenizer.texts_to_sequences(aspect_sequences)
    padded_aspects = pad_sequences(aspect_seqs, maxlen=maxlen, padding='post')
    return np.vstack(text_sequences), np.array(padded_aspects), np.array(labels)

train_texts, train_aspects, train_labels = preprocess_dataset(raw_train_ds, generate_aspect_terms(raw_train_ds))
val_texts, val_aspects, val_labels = preprocess_dataset(raw_val_ds, generate_aspect_terms(raw_val_ds))
test_texts, test_aspects, test_labels = preprocess_dataset(raw_test_ds, generate_aspect_terms(raw_test_ds))

print(f"train_texts shape: {train_texts.shape}")
print(f"train_aspects shape: {train_aspects.shape}")
print(f"Number of train_labels: {len(train_labels)}")
print(f"val_texts shape: {val_texts.shape}")
print(f"val_aspects shape: {val_aspects.shape}")
print(f"Number of val_labels: {len(val_labels)}")
print(f"test_texts shape: {test_texts.shape}")
print(f"test_aspects shape: {test_aspects.shape}")
print(f"Number of test_labels: {len(test_labels)}")

def build_model(vocab_size, maxlen, embedding_dim):
    text_inputs = Input(shape=(maxlen,))
    aspect_inputs = Input(shape=(maxlen,))
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    
    text_embedding = embedding_layer(text_inputs)
    aspect_embedding = embedding_layer(aspect_inputs)
    
    concatenated = Concatenate()([text_embedding, aspect_embedding])
    
    lstm = Bidirectional(LSTM(64, return_sequences=True))(concatenated)
    attention = Attention()([lstm, lstm])
    
    # Use Lambda layer for reduce_sum
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention)
    
    dense1 = Dense(64, activation='relu')(context_vector)
    outputs = Dense(1, activation='sigmoid')(dense1)
    
    model = Model(inputs=[text_inputs, aspect_inputs], outputs=outputs)
    return model

model = build_model(vocab_size, maxlen, embedding_dim)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


epochs = 10
batch_size = 32

history = model.fit([train_texts, train_aspects], train_labels, validation_data=([val_texts, val_aspects], val_labels), epochs=epochs, batch_size=batch_size)

loss, accuracy = model.evaluate([test_texts, test_aspects], test_labels)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


# experiment 1: Aspect Extraction Evaluation

# Evaluate the model's performance for aspect extraction
# Compute precision, recall, and F1-score for aspect extraction
from sklearn.metrics import precision_score, recall_score, f1_score

# Generate y_true for aspect extraction evaluation
y_true = (train_aspects > 0).flatten()  # Convert padded aspect sequences to binary labels

# Generate y_pred for aspect extraction evaluation
# Predict aspect terms using the trained model
y_pred = (model.predict([train_texts, train_aspects]) > 0.5).flatten()

# Ensure that y_true and y_pred have the same length
y_true = y_true[:len(y_pred)]


precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Aspect Extraction Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)



# Experiment 2: Aspect-Based Sentiment Classification

# Evaluate the model's performance for aspect-based sentiment classification
# Compute precision, recall, and F1-score for sentiment classification
from sklearn.metrics import classification_report

# Predict sentiment for test data
y_pred = (model.predict([test_texts, test_aspects]) > 0.5).astype("int32")

# Generate classification report
print("Aspect-Based Sentiment Classification Report:")
print(classification_report(test_labels, y_pred, target_names=['Negative', 'Positive']))


# experiment 3: hyperparameter tuning

# Define the parameter grids for testing
param_grids = [
    {
        'vocab_size': [10000],
        'maxlen': [None],  # Set maxlen to None for dynamic calculation
        'embedding_dim': [16],
        'lstm_units': [64],
        'dense_units': [64],
        'batch_size': [32]
    },
    {
        'vocab_size': [20000],
        'maxlen': [None],  # Set maxlen to None for dynamic calculation
        'embedding_dim': [32],
        'lstm_units': [128],
        'dense_units': [128],
        'batch_size': [64]
    }
]

results = []

for i, param_grid in enumerate(param_grids, start=1):
    print(f"Executing hyperparameter testing for set {i}...")
    
    # Calculate maxlen based on the actual length of the input sequences
    maxlen = train_texts.shape[1]

    # Build the model with current hyperparameters
    model = build_model(param_grid['vocab_size'][0], maxlen, param_grid['embedding_dim'][0])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit([train_texts, train_aspects], train_labels, 
                        validation_data=([val_texts, val_aspects], val_labels), 
                        epochs=epochs, batch_size=param_grid['batch_size'][0], verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate([val_texts, val_aspects], val_labels)

    # Log the results
    result = {
        'vocab_size': param_grid['vocab_size'][0],
        'maxlen': maxlen,  # Update maxlen here
        'embedding_dim': param_grid['embedding_dim'][0],
        'lstm_units': param_grid['lstm_units'][0],
        'dense_units': param_grid['dense_units'][0],
        'batch_size': param_grid['batch_size'][0],
        'validation_loss': loss,
        'validation_accuracy': accuracy
    }
    results.append(result)
    print(f'Hyperparameters for set {i}: {result}')

