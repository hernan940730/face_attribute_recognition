from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from keras.models import Model

import sys, getopt
import numpy as np
    
image_path = "cat.jpg"
weights_path = None
train_model = False

args_array = ["image_path=", "train_model="]

def load_args(argv):
    '''
    Load the arguments given by the user
    '''
    global image_path
    try:
        opts, args = getopt.getopt(argv, "i:t:", args_array)
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
        else:
            print ('Invalid option', argv)
            sys.exit(2)

def load_model(weights_path):
    '''
    Load the model for the Neural Network
    '''
    model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=None)
    
    return model

def predict():
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

if __name__ == "__main__":
    
    load_args(sys.argv[1:])
    
    print ("Loading model...")
    model = load_model()
    print ("Model loaded.")
    
    print ("Predicting...")
    preds = predict()
    print ('Predicted:', decode_predictions(preds, top=3)[0])
