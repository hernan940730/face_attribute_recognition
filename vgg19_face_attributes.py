from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from keras.layers import Flatten, Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

from my_utils import load_args, load_attributes, load_data

import cv2
import os

import sys
import numpy as np

HAAR_WEIGHTS_PATH = "hweights/"
BATCH_SIZE = 32

def load_model (weights_path = None, label_count = 40):
    '''
    Load the model for the Neural Network
    '''

    model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(224, 224, 3))
    
    print ("VGG19 Model:")
    model.summary()

    input_layer = Input(shape=(224, 224, 3), name = 'image_input')
    last_layer = model(input_layer)

    x = Flatten(name='flatten')(last_layer)
    output_layer = Dense(label_count, activation='sigmoid', name='predictions')(x)
    
    model = Model(input = input_layer, output = output_layer)
    
    if weights_path != None:
        model.load_weights(weights_path)
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print ("Our Model:")
    model.summary()
    plot_model(model, to_file='model.png')
    
    return model

def preprocess_image(image_path):
    
    face_class = os.path.join(HAAR_WEIGHTS_PATH, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_class)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if (len(faces) == 0):
        return cv2.resize(img, (224, 224))
    
    (x, y, w, h) = faces[0]
    
    eps_x = w / 2
    eps_y = h / 3
    
    img_height, img_width, img_channels = img.shape
    y_from = int(max(y - eps_y, 0))
    y_to = int(min(y + h + eps_y, img_height))
    x_from = int(max(x - eps_x, 0))
    x_to = int(min(x + w + eps_x, img_width))
    
    crop_img = img[y_from : y_to, x_from : x_to] 
    crop_img = cv2.resize(crop_img, (224, 224))
    print ("cropped img")
    cv2.imwrite("cropped.png", crop_img)
    return crop_img

def predict (image_path):
    img = preprocess_image(image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

def train(dataset_folder, session_folder, attributes, epochs = 100000, batch_size = 32):
    train_datagen = ImageDataGenerator(
        rescale=1./3.,
        fill_mode = "nearest",
        rotation_range=15,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./3.)

    (x_train, y_train, x_test, y_test) = load_data(dataset_folder, attributes)

    train_generator = train_datagen.flow (
        x_train, y_train,
        seed=732912,
        batch_size=batch_size
        )
    
    test_generator = test_datagen.flow (
        x_test, y_test,
        seed=2738,
        batch_size=batch_size
        )

    weights_folder = os.path.join(session_folder, 'weights')
    log_folder = os.path.join(session_folder, 'logs')

    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    tensorboard_callback = TensorBoard(log_dir=log_folder)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(weights_folder, "weights.{epoch:02d}.hdf5"), save_weights_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch= len(x_train) / batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(x_test) / batch_size,
        callbacks=[tensorboard_callback, checkpoint_callback])


if __name__ == "__main__":
    
    args_map = load_args( sys.argv[1:] )
    
    image_path = args_map["image_path"]
    weights_path = args_map["weights_path"]
    train_model = args_map["train_model"]
    
    dataset_path = "dataset/" if args_map["dataset_path"] == None else args_map["dataset_path"]
    session_path = "session/" if args_map["session_path"] == None else args_map["session_path"]
    
    epochs = 100000 if args_map["epochs"] == None else args_map["epochs"]
    batch_size = 32 if args_map["batch_size"] == None else args_map["batch_size"]
    
    attr = load_attributes(os.path.join(dataset_path, 'Anno/list_attr_celeba.txt'))
    label_count = attr["labels_count"]
    
    print ("Loading model...")
    model = load_model(weights_path)
    print ("Model loaded.")
    
    if (image_path != None):
        print ("Predicting...")
        preds = predict(image_path)
        sel_labels = []
        for i in range(len(preds[0])):
            if preds[0][i] == 1:
                sel_labels.append(attr["labels"][i])
        print ('Predicted:', sel_labels)
    if (train_model == True):
        print ("Training...")
        train(dataset_path, session_path, attr, epochs, batch_size)
        print ("Trained.")
        
