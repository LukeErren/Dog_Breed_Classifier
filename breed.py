# import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing import image  
import cv2                
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input #, decode_predictions
from keras.applications.resnet50 import ResNet50
import pickle

def Xception_predict_breed(img_path):
    # Extract the bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # Make prediction
    predicted_vector = Xception_model.predict(bottleneck_feature)
    breed = dog_names[np.argmax(predicted_vector)]
    # Make a more readable string
    breed = breed[15:].replace('_', ' ')
    return breed

def find_dog_breed_on_humans_and_dogs(image_location) :
    # Determen breed
    breed = Xception_predict_breed(image_location)
    
    # Determen human, dog or other
    if dog_detector(image_location) == True : # True if dog detected in image
        label = "This dog looks likes a %s " % breed 
    elif face_detector(image_location) == True :
        label = "This human looks likes a %s " % breed 
    else :
        label = "No dogs or humans detected" 
    
    img = cv2.imread(image_location)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.tick_params( axis='both', which='both', bottom=False, top=False,  left=False, right=False, labelleft=False,
                    labelbottom=False)
    plt.xlabel(label, fontsize=14 )
    plt.savefig('result.png', bbox_inches='tight', transparent=True )

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# Load variables
print ("Load dog names")
with open('data/dog_names.pkl', "rb") as fp:   
    dog_names = pickle.load(fp)    
print ("Load face cascade")
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
print ("Load model")
Xception_model = keras.models.load_model('data/trained_breed_model.pkl')