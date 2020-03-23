from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
from flask import Flask, jsonify
from flask import abort
from flask import request

app = Flask(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# TensorFlow and tf.keras
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array, img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    a = class_names[predicted_label]
    return a


@app.route('/file/<file>', methods=['GET'])
def upload_file(file):
    fr = '/Users/a912158001/Desktop/' + file

    def anal(fr):
        model = keras.models.load_model(
            '/Users/a912158001/PycharmProjects/image_classification/my_model.h5')
        test_images = Image.open(fr)
        test_images = np.invert(test_images.convert('L'))
        test_images = test_images / 255.0
        probability_model = tf.keras.Sequential([model,
                                                 tf.keras.layers.Softmax()])
        test_images = (np.expand_dims(test_images, 0))
        predictions_single = probability_model.predict(test_images)
        b = plot_image(0, predictions_single[0], test_images)
        return b
    return anal(fr)








