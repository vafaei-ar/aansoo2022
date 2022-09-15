
import os
import streamlit as st
from PIL import Image
from tensorflow import keras

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

st.write('This is our application, it is running on streamlit')

# uploaded_file = st.file_uploader("Please ulpoad yout image", type="jpg")
# print(uploaded_file)

import numpy as np

model = keras.models.load_model('models/model_v1.h5')


class_names = ['Cat', 'Dog']

image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    # st.write(file_details)
    img = load_image(image_file)
    img = img.resize((224,224))
    st.image(img,width=250)
    with open(os.path.join("tempDir",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())      
      
    img = np.array(img)/255
    out = model(img[None,:,:,:])
    # class_name =  np.random.choice(class_names) 
    class_name =  class_names[np.argmax(out)]   

    st.write(f'This is a {class_name} image')
      
         
    # st.success("Saved File")