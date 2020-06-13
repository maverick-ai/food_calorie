import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
#from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
import pandas as pd


model=tf.keras.models.load_model("/Users/sushmabansal/Desktop/trained_food_new1.h5")
img_path = "/Users/sushmabansal/Desktop/food_data/images/apple_pie/134.jpg"
img = image.load_img(img_path, target_size=(360, 360))
x = image.img_to_array(img)/255
x = np.expand_dims(x, axis=0)
outputclass= model.predict(x)

#model=tf.keras.models.load_model("pretrained_food.hdf5",compile=False)

#img_path = "/Users/sushmabansal/Desktop/food_data/images/apple_pie/134.jpg"
#img = image.load_img(img_path, target_size=(299, 299))
#img_processed = preprocess_input(img)
#imgs = np.expand_dims(img_processed, 0)
#outputclass= model.predict(imgs)
#data=pd.read_csv("/Users/sushmabansal/Desktop/food.csv", sep="\t", low_memory=False)