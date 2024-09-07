from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import keras
from tensorflow.keras import backend as K 
import io
import os
import cv2
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import base64
import matplotlib.pyplot as plt
from PIL import Image
from ipywidgets import FileUpload
from IPython.display import display

from tensorflow.keras.preprocessing.image import img_to_array, load_img
app = Flask(__name__)

smooth = 1e-15
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.reshape(y_pred, (-1, 7)) # Reshape to (num_pixels, 7)
    y_pred_f = K.one_hot(K.argmax(y_pred_f, axis=-1), num_classes=7) # Convert to one-hot
    y_pred_f = K.flatten(y_pred_f)
    
    # Reshape y_true_f to match the shape of y_pred_f
    y_true_f = K.one_hot(K.cast(y_true_f, 'int32'), num_classes=7) # Convert to one-hot
    y_true_f = K.flatten(y_true_f)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + 1.0) / (union + 1.0)
segmentation_model=tf.keras.models.load_model(r"seg.h5", custom_objects={'jacard_coef':                   
jacard_coef},compile=False)

save_image_path='mini code'
def preprocess_mask(mask_path, num_classes=7):
    mask = cv2.imread(mask_path)

    if mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    encoded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    class_colors = [
        [250, 235, 185],
        [200, 200, 200],
        [100, 100, 150],
        [80, 140, 50],
        [200, 160, 40],
        [40, 120, 240],
        [0, 0, 0]
    ]

    for class_id, color in enumerate(class_colors):
        class_mask = np.all(mask == color, axis=-1)
        encoded_mask[class_mask] = class_id

    return encoded_mask
def decode_mask(mask, label_mapping):
    # Initialize an empty array for the decoded mask
    decoded_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Loop through the label mapping and assign colors to corresponding class indices
    for label, color in label_mapping.items():
        decoded_mask[mask == label] = color

    return decoded_mask
label_mapping = {
    (250, 235, 185): 0,  # Informal Settlements
    (200, 200, 200): 1,  # Built-Up
    (100, 100, 150): 2,  # Impervious Surfaces
    (80, 140, 50): 3,    # Vegetation
    (200, 160, 40): 4,   # Barren
    (40, 120, 240): 5,   # Water
    (0, 0, 0): 6         # Unlabelled
}

# Inverse mapping to convert class indices back to RGB values
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
def preprocess_single_image(image_path, target_size=(128, 128,3)):
    
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    
    return np.expand_dims(img_array, axis=0)
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.reshape(y_pred, (-1, 7)) # Reshape to (num_pixels, 7)
    y_pred_f = K.one_hot(K.argmax(y_pred_f, axis=-1), num_classes=7) # Convert to one-hot
    y_pred_f = K.flatten(y_pred_f)
    
    # Reshape y_true_f to match the shape of y_pred_f
    y_true_f = K.one_hot(K.cast(y_true_f, 'int32'), num_classes=7) # Convert to one-hot
    y_true_f = K.flatten(y_true_f)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + 1.0) / (union + 1.0)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # Read the image file as binary
        image_file = request.files['image']
        filename=secure_filename(image_file.filename)
        
        image_binary = image_file.read()
        
        # Convert the binary data to numpy array
        nparr = np.frombuffer(image_binary, np.uint8)
        
        # Decode the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image1=image
        image_path=os.path.join(r"uploads",filename)
      
        image = preprocess_single_image(image_path)

        # Predict the mask
        predicted_mask = segmentation_model.predict(image)[0]
        predicted_mask = np.argmax(predicted_mask, axis=-1)
        predicted_mask = decode_mask(predicted_mask, inverse_label_mapping)

        # Convert images to base64 for displaying in HTML
        input_buffer = io.BytesIO()
        output_buffer = io.BytesIO()
        input_pil_image = Image.fromarray(image1)
        output_pil_image = Image.fromarray(predicted_mask)
        input_pil_image.save(input_buffer, format='png')
        output_pil_image.save(output_buffer, format='png')
        input_image = base64.b64encode(input_buffer.getvalue()).decode('utf-8')
        mask_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        # Render the index template with the input and output images
        return render_template('index.html', input_image=input_image, mask_image=mask_image,prediction_text="Segmentation Completed!!!")

    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
