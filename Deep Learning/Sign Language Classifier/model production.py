import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

model = load_model("98,97 Sign Language ALS Classifier.h5")

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'del', 'nothing', 'space']

from keras.preprocessing import image
img = image.load_img("j.jpg", target_size=(224, 224, 3))
img = image.img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# Get probability of each letter
result_probabilities = model.predict([img])
# Select the highest probability index, since it's the answer (letter)
letter_index = np.argmax(result_probabilities)
letter = labels[letter_index]

print(letter)
# return letter