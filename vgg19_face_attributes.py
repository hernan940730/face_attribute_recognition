from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from keras.layers import Flatten, Input, Dense
from keras.models import Model
from keras.utils import plot_model

from my_utils import load_args
from my_utils import load_attributes

import cv2
import os

import sys
import numpy as np

DATASET_FOLDER="dataset/"
HAAR_WEIGHTS_FOLDER="hweights/"


    
labels = ["eyes"] * 3

def load_model (weights_path = None):
    '''
    Load the model for the Neural Network
    '''

    model = VGG19(include_top=False,  input_tensor=None, input_shape=(224, 224, 3))
    
    print ("VGG19 Model:")
    model.summary()

    input_layer = Input(shape=(224,224,3), name = 'image_input')
    last_layer = model(input_layer)

    x = Flatten(name='flatten')(last_layer)
    output_layer = Dense(40, activation='sigmoid', name='predictions')(x)
    
    model = Model(input = input_layer, output = output_layer)
    
    if weights_path != None:
        model.load_weights(weights_path)
        
    print ("Our Model:")
    model.summary()
    plot_model(model, to_file='model.png')
    
    return model


def preprocess_image(image_path):
    face_class = os.path.join(HAAR_WEIGHTS_FOLDER, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_class)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    (x, y, w, h) = faces[0]
    crop_img = img[y:y + h, x:x + w] 
    crop_img = cv2.resize(crop_img, (224, 224)) 
    print ("cropped img")
    cv2.imwrite("cropped.png", crop_img)
    return crop_img
    


def predict (image_path, model):
    img = preprocess_image(image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

def train ():
    pass

if __name__ == "__main__":
    
    attr = load_attributes(os.path.join(DATASET_FOLDER, 'Anno/list_attr_celeba.txt'))
    
    print ("Image count:", attr["img_count"])
    print ("Label count:", attr["labels_count"])
    print (attr["attributes"][0])
    
    args_map = load_args( sys.argv[1:] )
    
    if args_map["dataset_path"] != None:
        DATASET_FOLDER = args_map["dataset_path"]
    
    image_path = args_map["image_path"]
    weights_path = args_map["weights_path"]
    train_model = args_map["train_model"]
    
    print ("Loading model...")
    model = load_model(weights_path)
    print ("Model loaded.")
    
    if (image_path != None):
        print ("Predicting...")
        preds = predict(image_path, model)
        print ('Predicted:', preds)
        print (sum(preds[0]))
    if (train_model == True):
        print ("Training...")
            
