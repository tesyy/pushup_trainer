import pprint
import pandas as pd
import tensorflow as tf
from data_processing.create_data_matrices_ver1 import get_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from tensorflow.keras.regularizers import l2
# from pycm import *
# import seaborn as sns

USE_MODEL = False



if not USE_MODEL:
    #av_testAcc = 0
    #test_neighbor = [20,22,24,26,28,30,32,34,36]
    #neighbor_acc = []
    neighbor = 6
    input, output = get_data("train",neighbor)
    input, output = shuffle(input, output, random_state=42)
    test_features, test_labels = get_data("test",neighbor)

    # Reshape the input data for LSTM: 
    # Convert from 2D (samples, features) to 3D (samples, time_steps, features)
    # Assuming time_steps = 1 (single step per sample), reshape input data to have (samples, 1, features)
    input = np.reshape(input, (input.shape[0], 1, input.shape[1]))  # (samples, time_steps, features)
    test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))  # (samples, time_steps, features)

    split = int(0.8 * len(input))
    (train_features, train_labels), (val_features, val_labels) = (input[:split], output[:split]), \
                                                                (input[split:], output[split:])

    count = 0###################
    for i in range(0,1):
        model = tf.keras.Sequential([
            # LSTM layer with 40 units, and input shape should be (time_steps, features)
            tf.keras.layers.LSTM(40, activation='tanh', input_shape=(train_features.shape[1], train_features.shape[2])),
            tf.keras.layers.BatchNormalization(),
            # Dropout 10%
            #tf.keras.layers.Dropout(0.1),
            #add a l2 regulation layer
            tf.keras.layers.Dense(128, kernel_regularizer=l2(0.003), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # Dense layer after LSTM
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['accuracy'],
                    optimizer=opt)

        hist = model.fit(train_features, train_labels, epochs=180, validation_data=(val_features, val_labels))

        acc = hist.history['accuracy']
        loss = hist.history['loss']
        val_acc = hist.history['val_accuracy']
        val_loss = hist.history['val_loss']
        test_loss, test_acc = model.evaluate(test_features,test_labels)
        #av_testAcc += test_acc #########
        #count += 1##############
        print(f"Validation Loss: {val_loss[-1]}\nValidation Accuracy: {val_acc[-1]}")
        print(f"Test Loss: {test_loss}\nTest Accuracy: {test_acc}")
        model.save("working_model_5.keras")

        plt.plot(acc,'b-', label='Training Accuracy')
        plt.plot(val_acc, 'r-', label='Validation Accuracy')
        plt.title('Training/Validation Accuracy')
        plt.legend()

        plt.figure()

        plt.plot(loss,"b-", label='Training Loss')
        plt.plot(val_loss, 'r-', label='Validation Loss')
        plt.title('Training/Validation Loss')
        plt.legend()
        #print(f"finished {i+1} rounds")
        #plt.show()
    #av_testAcc = av_testAcc/count
    #neighbor_acc.append(av_testAcc)
    #print(f"average test accuracy is: {av_testAcc}")
    #for i in range(len(neighbor_acc)):
    #    print(neighbor_acc[i])
else:
    model = tf.keras.models.load_model("working_model_4.keras")
    preds = model.predict(test_features)

    cm = confusion_matrix(actual_vector=test_labels[0], predict_vector=preds[0])
    print(cm.table)
