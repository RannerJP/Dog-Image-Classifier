import os
import urllib.request
from tkinter import Tk
from dogclassification.dog_classification_UI import DogClassificationUI
from keras.models import load_model

if __name__ == '__main__':
    try:
        model = load_model(os.path.join('models', 'DogClassification.h5'))
    except OSError:
        file = urllib.request.urlretrieve("https://github.com/RannerJP/Dog-Image-Classifier/raw/main/models/DogClassification.h5?download=", ".h5")
        model = load_model(file[0])    
    window = Tk()
    user_interface = DogClassificationUI(window, model)
    user_interface.show_UI()