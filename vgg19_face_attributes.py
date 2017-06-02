from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from keras.layers import Flatten, Input, Dense
from keras.models import Model
from keras.utils import plot_model

from my_utils import load_args

import sys
import numpy as np
    
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
    x = Dense(4096, activation='relu', name='fc1')(x)
    
    output_layer = Dense(40, activation='sigmoid', name='predictions')(x)
    
    model = Model(input = input_layer, output = output_layer)
    
    if weights_path != None:
        model.load_weights(weights_path)
        
    print ("Our Model:")
    model.summary()
    plot_model(model, to_file='model.png')
    
    return model

def predict (image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

def train ():
    pass

if __name__ == "__main__":
    
    args_map = load_args( sys.argv[1:] )
    
    image_path = args_map["image_path"]
    weights_path = args_map["weights_path"]
    train_model = args_map["train_model"]
    
    print ("Loading model...")
    model = load_model(weights_path)
    print ("Model loaded.")
    
    if (image_path != None):
        print ("Predicting...")
        preds = predict(image_path)
        print ('Predicted:', preds)
        print (sum(preds[0]))
    if (train_model == True):
        print ("Training...")
            
