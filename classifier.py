import streamlit as st
import base64
from pandas import read_csv
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras import ops
from PIL import Image
import numpy as np
import cv2
import dill


st.markdown('<h1 style="color:black;">Forest Amazon Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model is able to recognize the following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;">17 classes of atmospheric conditions, common and rare Land Uses and Land Covers</h3>', unsafe_allow_html=True)

# Background image

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    [data-testid="stVerticalBlockBorderWrapper"]> div:first-child{
    background-color: #f0f2f6;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('Img_bg.jpg')

# Upload image

upload = st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2 = st.columns(2)
if upload is not None:
  im = Image.open(upload)
  imgg = np.asarray(im)
  image = cv2.resize(imgg, (224, 224))
  imgg = preprocess_input(image)
  imgg = np.expand_dims(imgg, 0)
  c1.header('Input Image')
  c1.image(im)
  #c1.write(imgg.shape)

# Import Tags + model weights

with open('lulc3_model.pkl', 'rb') as f:
    inv_mapping = dill.load(f)
    mapping_csv = dill.load(f)

model = load_model('final_model.h5')

# Make a prediction

# create a mapping of tags to integers given the loaded mapping file
def create_tag_mapping(mapping_csv):
	labels = set()
	for i in range(len(mapping_csv)):
		tags = mapping_csv['tags'][i].split(' ')
		labels.update(tags)
	labels = list(labels)
	labels.sort()
	labels_map = {labels[i]:i for i in range(len(labels))}
	inv_labels_map = {i:labels[i] for i in range(len(labels))}
	return labels_map, inv_labels_map

# convert a prediction to tags
def prediction_to_tags(inv_mapping, prediction):
	values = prediction.round()
	tags = [inv_mapping[i] for i in range(len(values)) if values[i] == 1.0]
	return tags

# load an image and predict the class
def run_example(inv_mapping):
    img = load_img(upload, target_size=(128, 128))
    img = img_to_array(img)
    img = img.reshape(1, 128, 128, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    result = model.predict(img)
    print(result[0])
    tags = prediction_to_tags(inv_mapping, result[0])
    return tags

# create a mapping of tags to integers
_, inv_mapping = create_tag_mapping(mapping_csv)

# Prediction

c2.header('Output')
c2.subheader('Predicted class :')
if upload is not None:
    lst = run_example(inv_mapping)
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    c2.markdown(s)
