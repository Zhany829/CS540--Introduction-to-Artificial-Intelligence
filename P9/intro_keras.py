# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:27:10 2020

@author: yuzha
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_dataset(training=True):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training is True:
        return np.array(train_images),np.array(train_labels)
    else:
        return np.array(test_images),np.array(test_labels)
    
def print_stats(train_images, train_labels):
    class_names = [
        "Zero",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
    ]
    cnt = {}
    for name in class_names:
        cnt[name] = 0
    for label in train_labels:
        cnt[class_names[label]] += 1
    print((len(train_images)))
    print(("{}x{}".format(len(train_images[0]), len(train_images[0][0]))))

    for i in range(len(class_names)):
        print(("{}. {} - {}".format(i, class_names[i], cnt[class_names[i]])))

def build_model():
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10))
    op = keras.optimizers.SGD(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=op, loss=loss_fn, metrics=["accuracy"])
    return model


def train_model(model, train_images, train_labels, T):
    model.fit(x=train_images, y=train_labels, epochs=T)


def evaluate_model(model, test_images, test_labels, show_loss=True):
    s = model.evaluate(test_images, test_labels, verbose=0)
    s[1] = s[1] * 100
    if show_loss == True:
        print(("Loss: {}".format("%.4f" % s[0])))
    print(("Accuracy: {}%".format("%.2f" % s[1])))


def predict_label(model, test_images, index):
    predicted = model.predict(test_images)
    result = predicted[index]
    labels = []
    values = []

    temp_1 = 0
    ind_1 = -1
    temp_2 = 0
    ind_2 = -1
    temp_3 = 0
    ind_3 = -1
    for j in range(len(result)):
        if result[j] > temp_1:
            temp_1 = result[j]
            ind_1 = j
        elif temp_1 > result[j] > temp_2:
            temp_2 = result[j]
            ind_2 = j
        elif temp_2 > result[j] > temp_3:
            temp_3 = result[j]
            ind_3 = j

    values.append(temp_1)
    values.append(temp_2)
    values.append(temp_3)
    labels.append(ind_1)
    labels.append(ind_2)
    labels.append(ind_3)

    for i in range(len(labels)):
        if labels[i] == 0:
            print(("Zero: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 1:
            print(("One: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 2:
            print(("Two: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 3:
            print(("Three: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 4:
            print(("Four: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 5:
            print(("Five: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 6:
            print(("Six: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 7:
            print(("Seven: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 8:
            print(("Eight: " + str("{:.2%}".format(values[i]))))
        elif labels[i] == 9:
            print(("Nine: " + str("{:.2%}".format(values[i]))))
