from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from keras.layers import Flatten, Input, Dense
from keras.models import Model
from keras.utils import plot_model

import sys, getopt
import numpy as np
    
image_path = "cat.jpg"
weights_path = None
train_model = False

args_array = ["image_path=", "train_model=", "weights_path="]

labels = ["eyes"] * 3

def load_args(argv):
    '''
    Load the arguments given by the user
    '''
    global image_path
    global weights_path
    
    try:
        opts, args = getopt.getopt(argv, "i:t:w:", args_array)
    except getopt.GetoptError:
        print ('Invalid option', argv)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--image_path"):
            image_path = arg
        elif opt in ("-t", "--train_model"):
            if arg in (True, False):
                train_model = arg
            else:
                print ('Invalid option', opt, arg)
                sys.exit(2)
        elif opt in ("-w", "--weights_path"):
            weights_path = arg
        else:
            print ('Invalid option', argv)
            sys.exit(2)

def load_model(weights_path = None):
    '''
    Load the model for the Neural Network
    '''
    
    model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(224, 224, 3))
    model.summary()
    
    input_layer = Input(shape=(224,224,3), name = 'image_input')
    last_layer = model(input_layer)
    
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    output_layers = ( 
        Dense(3, activation='softmax', name='eye_predictions')(x), 
        Dense(5, activation='softmax', name='hair_predictions')(x), 
        Dense(2, activation='softmax', name='face_predictions')(x)
        )
    
    model = Model(input = input_layer, output = output_layers)
    
    if weights_path != None:
        model.load_weights(weights_path)
    model.summary()
    plot_model(model, to_file='model.png')
    
    
    return model

def predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

if __name__ == "__main__":
    print(len(labels))
    load_args(sys.argv[1:])
    
    print ("Loading model...")
    model = load_model(weights_path)
    print ("Model loaded.")
    
    print ("Predicting...")
    preds = predict(image_path)
    print ('Predicted:', preds)
