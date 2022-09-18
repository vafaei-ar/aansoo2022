
import os
import mlflow
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pylab as plt
import ktrans as ktr

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

tab1, tab2, tab3 = st.tabs(["Train", "Prediction", "About Me"])

with tab1:
  st.title("Train")
  img_height, img_width = 224, 224
  batch_size = 32

  # data_dir = '/home/gf/works/datasets/dogs-vs-cats/train_dired/'
  data_dir = st.text_input('Data path', '/home/gf/works/datasets/dogs-vs-cats/train_dired/')
      
  if st.button('Train') and data_dir:
    
    mlflow.start_run()
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      label_mode = 'categorical',
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      label_mode = 'categorical',
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
   
    base_model = tf.keras.applications.VGG16(input_shape=(img_height, img_width, 3),
                                             weights='imagenet', include_top=False)
    
    base_model.trainable = False
    x_in = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    x_conv = base_model(x_in, training=False)
    x_out = tf.keras.layers.GlobalAveragePooling2D(name='GlobalAVG')(x_conv)
    x_pred = tf.keras.layers.Dense(2)(x_out)
    
    model = tf.keras.Model(inputs=x_in, outputs=x_pred)
      
    model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()])
    
    hist = model.fit(train_ds, validation_data=val_ds, epochs=2)
    model_path = 'models/model_v1.h5'
    model.save(model_path)
    
    
    metrics = {'loss':hist.history['loss'],
               'val_loss':hist.history['val_loss'],
               'binary_accuracy':hist.history['binary_accuracy'],
               'val_binary_accuracy':hist.history['val_binary_accuracy'],   
    }
    mlflow.log_metrics(metrics)
    
    tags = {'model_path': model_path,
            'Architecture': base_model.name
            }
    mlflow.set_tags(tags)
    
    mlflow.end_run()
    
    st.write('Done')
    
    
with tab2:
   
  st.write('This is our application, it is running on streamlit')

  model_path = 'models/model_v1.h5'
  model = keras.models.load_model(model_path)

  class_names = ['Cat', 'Dog']

  image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
  if image_file is not None:
      file_details = {"FileName":image_file.name,"FileType":image_file.type}
      # st.write(file_details)
      img = load_image(image_file)
      img = img.resize((224,224))
      img = np.array(img)/255
      img = img[None,:,:,:]
      out = model(img)
      # class_name =  np.random.choice(class_names) 
      class_name =  class_names[np.argmax(out)]   
      
      if st.checkbox('Saliancy Map'):
        idx = np.argmax(out[0].numpy())
        smap = ktr.vanilla_saliency(img,model,class_id=np.array([idx==0,idx==1]).astype(int))
        fig,ax = plt.subplots(1,1,figsize=(4,4))
        ax.imshow(img[0])
        ax.imshow(smap[0],cmap='jet',alpha=0.3)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('interp.jpg')    
        plt.close()  
        st.image('interp.jpg')
      else:
        st.image(img)
      with open(os.path.join("tempDir",image_file.name),"wb") as f: 
        f.write(image_file.getbuffer())      
        

      st.write(f'This is a {class_name} image')
        
          



          
with tab3:
  st.title("About Me")
  st.write("This is the about me tab")