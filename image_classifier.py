import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images

training_images, testing_images = training_images/255, testing_images/255

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:20000]
testing_labels= testing_labels[:20000]

models = models.load_model('image.h5')

img = cv.imread('plane.jpeg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

predictions = models.predict(np.array([img])/255)
index = np.argmax(predictions)
print(predictions)
print(f'Prediction is {class_names[index]}')