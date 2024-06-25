import os
from tkinter import Tk
from DogClassificationUI import DogClassificationUI
from keras.models import load_model

if __name__ == '__main__':
    model = load_model(os.path.join('models', 'DogClassification.h5'))
    window = Tk()
    user_interface = DogClassificationUI(window, model)
    user_interface.show_UI()