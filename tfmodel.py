import pprint
import pandas as pd
import tensorflow as tf
from data_processing.create_data_matrices import get_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
# from pycm import *
# import seaborn as sns

USE_MODEL = False

input, output = get_data()
input, output = shuffle(input, output, random_state=42)

# Reshape the input data for LSTM: 
# Convert from 2D (samples, features) to 3D (samples, time_steps, features)
# Assuming time_steps = 1 (single step per sample), reshape input data to have (samples, 1, features)
input = np.reshape(input, (input.shape[0], 1, input.shape[1]))  # (samples, time_steps, features)

split = int(0.8 * len(input))
(train_features, train_labels), (test_features, test_labels) = (input[:split], output[:split]), \
                                                               (input[split:], output[split:])

if not USE_MODEL:

    model = tf.keras.Sequential([
        # LSTM layer with 50 units, and input shape should be (time_steps, features)
        tf.keras.layers.LSTM(50, activation='tanh', input_shape=(train_features.shape[1], train_features.shape[2])),
        
        # Dense layer after LSTM
        tf.keras.layers.Dense(4, activation='relu'),
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  optimizer=opt)

    hist = model.fit(train_features, train_labels, epochs=200)

    acc = hist.history['accuracy']
    loss = hist.history['loss']

    valid_loss, valid_acc = model.evaluate(test_features, test_labels)

    print(f"Validation Loss: {valid_loss}\nValidation Accuracy: {valid_acc}")

    model.save("working_model_1.keras")

    plt.plot(acc, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(loss, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.show()
else:
    model = tf.keras.models.load_model("working_model_1.keras")
    preds = model.predict(test_features)

    cm = confusion_matrix(actual_vector=test_labels[0], predict_vector=preds[0])
    print(cm.table)
