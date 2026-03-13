import os
import numpy as np
from PIL import Image
import cv2

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

base_model = VGG19(include_top = False, input_shape = (224, 224,3))


x = base_model.output

flat = Flatten()(x)

class_1 = Dense(4608, activation = 'relu')(flat)

dropout = Dropout(0.2)(class_1)

class_2 = Dense(1152, activation='relu')(dropout)

output = Dense(2, activation='softmax')(class_2)

#model with empty weights
model_03 = Model(base_model.inputs, output)

#trained model
model_03.load_weights('vgg_unfrozen.h5')

#create flask app

app = Flask(__name__)

print("Model loaded. check localhost")

#========== Define Flask API =====

#Helper functions

def get_className(classNo):
    if classNo == 0:
        return "Normal"
    return "Pneumonia"

def getResult(img):

    #read the image
    image = cv2.imread(img)

    #Convert to RGB
    image = Image.fromarray(image, 'RGB')

    #Resize to the model's expected input size
    image = image.resize((224,224))

    #Convert the Numpy Array
    image = np.array(image)

    #Normalize the pixels values to [0,1]
    image = image/225.0

    #(1,224, 224, 3) 
    #Expand dimensions to add the batch size
    input_img = np.expand_dims(image, axis=0)


    #Make Prediction
    result = model_03.predict(input_img)


    #result["0.99", "0.01"]

    #Get the class index with the highest probability

    result01 = np.argmax(result, axis=1)

    return result01

@app.route('/', method = ['GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']

            base_path = os.path.dirname(__file__)
            
            file_path = os.path.join(

            base_path, 'uploads', secure_filename(f.filename)

            )
            f.save(file_path)
            
            value = getResult(file_path)
            
            result = get_className(value)
            return result
        
        except Exception as e:
            print(f"Error: {e}")
    return None
        



#==================

if __name__ == '__main__':
    app.run(debug=True)