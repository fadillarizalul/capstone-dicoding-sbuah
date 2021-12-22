import tensorflow as tf
import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('model.h5')
  return model
  
with st.spinner('Model is being loaded..'):
  model = load_model()

st.write("""
         # sBuah: Fruit Quality Classification
         """
         )
st.text('made by CSD-069 Team')
         
def predict_class(image, model):
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [150, 150])
	image = np.expand_dims(image, axis = 0)
	prediction = model.predict(image)
	return prediction

model = load_model()

st.text('This is a Web to Classify Fruit')
st.text('The result is whether the fruit is Fresh or Rotten')

file = st.file_uploader("Upload an image of a fruit", type=["jpg", "png"])

if file is None:
	st.text('Waiting for upload....')
  
else:
	slot = st.empty()
	slot.text('Running inference....')
	test_image = Image.open(file)
	st.image(test_image, caption="Input Image", width = 400)
	pred = predict_class(np.asarray(test_image), model)
	class_names = ['Fresh Banana', 'Fresh Mango', 'Fresh Orange',
					'Rotten Banana', 'Rotten Mango', 'Rotten Orange']
	result = class_names[np.argmax(pred)]
	output = 'The image is a ' + result
	slot.text('Image has successfully uploaded')
	st.success(output)
