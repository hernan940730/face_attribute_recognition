from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from keras.models import Model

import sys, getopt
import numpy as np
    
image_path = "cat.jpg"
args_array = ["image_path="]

def configure_args(argv):
    global image_path
    try:
        opts, args = getopt.getopt(argv, "i:", args_array)
    except getopt.GetoptError:
        print ('Invalid option', argv)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--image_path"):
            image_path = arg
        else:
            print ('Invalid option', argv)
            sys.exit(2)
            
if __name__ == "__main__":
    configure_args(sys.argv[1:])
    
    print ("Loading model...")
    model = VGG19(weights='imagenet')
    print ("Model loaded.")
    
    print ("Loading image...")
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print ("Image loaded.")
    print ("Predicting...")
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
