import os
from tkinter import Tk
import unittest
from dogclassification.dog_classification_UI import DogClassificationUI
import urllib.request
from keras.models import load_model
from PIL import Image
import time

class TestDogClassificationUI(unittest.TestCase):

    
    def setUp(self) -> None:
        try:
            self.valid_model = load_model(os.path.join('models', 'DogClassification.h5'))
            self.invalid_model = load_model(os.path.join('models', 'failModel.h5'))
        except OSError:
            file = urllib.request.urlretrieve("https://github.com/RannerJP/Dog-Image-Classifier/raw/main/models/DogClassification.h5?download=", ".h5")
            fail_model = urllib.request.urlretrieve("https://github.com/RannerJP/Dog-Image-Classifier/raw/main/models/failModel.h5?download=", ".h5")
            self.valid_model = load_model(file[0])
            self.invalid_model = load_model(fail_model[0])
        self.UI = DogClassificationUI(Tk(), self.valid_model)
    
    def tearDown(self) -> None:
         self.UI.window.destroy()
    def test_invalid_image(self):
        image = os.path.join('assets', 'Invalid_Image.jpg')
        self.UI.show_image(image)
        start_time = time.time()
        while time.time() - start_time < 3:
            self.UI.window.update_idletasks()
            self.UI.window.update()
        classification_type = self.UI.classification_text.cget("text")
        self.assertTrue(classification_type.startswith("Sorry, that image type is invalid!"))
    def test_valid_image_classifying_text(self):
        image = Image.open(os.path.join('assets', 'Valid_Image.jpg'))
        self.UI.show_image(image)
        start_time = time.time()
        while time.time() - start_time < 3:
            self.UI.window.update_idletasks()
            self.UI.window.update()
        classification_type = self.UI.classification_text.cget("text")
        self.assertTrue(classification_type.startswith("Your image is of a:"))