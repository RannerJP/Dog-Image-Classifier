import os
import sys
from tkinter import Tk
from dogclassification.dog_classification_UI import DogClassificationUI
from keras.models import load_model

if __name__ == '__main__':
    model = load_model(os.path.join('models', 'DogClassification.h5'))
    window = Tk()
    user_interface = DogClassificationUI(window, model)
    user_interface.show_UI()
    print(sys.path)