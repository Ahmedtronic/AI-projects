import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

def preprocess(image):
    image = Image.open(image)
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    image = image/255.0
    return image
