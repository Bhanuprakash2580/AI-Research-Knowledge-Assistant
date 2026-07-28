import os
import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
import numpy as np

MODEL_PATH = os.getenv("TF_CLASSIFIER_PATH", "models/classifier.h5")
SAVE_DIR = pathlib.Path(MODEL_PATH).parent
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(categories=None, num_samples=5000):
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers','footers','quotes'))
    texts = data.data
    targets = data.target
    return train_test_split(texts, targets, test_size=0.2, random_state=42)


def build_model(vocab_size=20000, sequence_length=500, num_classes=20):
    vectorize_layer = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = vectorize_layer(inputs)
    x = layers.Embedding(vocab_size, 64)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, vectorize_layer


def train_and_save(num_classes=20):
    X_train, X_test, y_train, y_test = prepare_data()
    model, vectorize_layer = build_model(num_classes=num_classes)
    # adapt vectorizer
    vectorize_layer.adapt(X_train)
    # fit model
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))
    # attach vectorizer for saving: create a wrapper model
    inp = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorize_layer(inp)
    x = model.layers[3](x) if len(model.layers) > 3 else x
    # Instead, save the model and vectorizer separately (user can adapt)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == '__main__':
    train_and_save()
