import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import logging
import json
import argparse

parser = argparse.ArgumentParser(description='My image classifier model')

parser.add_argument('input', action='store', type=str, help='Enter Image path')
parser.add_argument('model', action='store', type=str, help='Enter classifer path')
parser.add_argument('--top_k', default=5, action='store', type=int, help='Get the top k most likely classes')
parser.add_argument('--category_name', default='./label_map.json', action='store', type=str,
                    help='JSON file with labels')
arg_p = parser.parse_args()
top_k = arg_p.top_k


def process_image(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    img /= 255
    return img


def predict(image_path, model, top_k):
    image_path = image_path
    im = Image.open(image_path)
    test_image = np.asarray(im)
    vals = model.predict(np.expand_dims(process_image(test_image), axis=0))

    values, ind = tf.nn.top_k(vals, k=top_k)

    # conerting to list and storing values
    probs = list(values.numpy()[0])
    classes = list(ind.numpy()[0])
    return probs, classes


with open(arg_p.category_name, "r") as file:
    mapping = json.load(file)

loaded_model = tf.keras.models.load_model(arg_p.model, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

print(f'\n Top {top_k} Classes \n')
probs, labels = predict(arg_p.input, loaded_model, top_k)

for prob, label in zip(probs, labels):
    print('Label:', label)
    print('Class name', mapping[str(label + 1)].title())
    print('Probability:', prob)

##$ python predict.py /path/to/saved_model
