import os
import unittest
from dogclassification.image_classifier import ImageClassifier
from dogclassification.image_classifier import IncorrectOutputShapeError
from PIL import Image
from keras.models import load_model
import urllib.request


class TestImageClassifier(unittest.TestCase):
    def setUp(self):    
            try:
                self.valid_model = load_model(os.path.join('models', 'DogClassification.h5'))
            except OSError:
                file = urllib.request.urlretrieve("https://github.com/RannerJP/Dog-Image-Classifier/raw/main/models/DogClassification.h5?download=", ".h5")
                self.valid_model = load_model(file[0])
            self.classifier = ImageClassifier(self.valid_model)
    
    def test_valid_model_and_image(self):
        self.classifier.set_model(self.valid_model)
        image = Image.open(os.path.join('assets', 'Valid_Image.jpg'))
        image = image.resize((256,256))
        prediction = self.classifier.get_prediction(image)
        self.assertIn(prediction, [-1] + list(range(1,11)))
    
    def test_invalid_model_init(self):
        with self.assertRaises(TypeError):
            fail_classifier = ImageClassifier(None)
    
    def test_invalid_model_set(self):
        with self.assertRaises(TypeError):
            self.classifier.set_model(None)
    
    def test_invalid_model_output_shape_init(self):
        try:
            invalid_model = load_model(os.path.join('models', 'failModel.h5'))
        except OSError:
            file = urllib.request.urlretrieve("https://github.com/RannerJP/Dog-Image-Classifier/raw/main/models/failModel.h5?download=", ".h5")
            invalid_model = load_model(file[0])
        with self.assertRaises(IncorrectOutputShapeError):
            classifier = ImageClassifier(invalid_model)
            
    def test_invalid_model_output_shape_set(self):
        try:
            invalid_model = load_model(os.path.join('models', 'failModel.h5'))
        except OSError:
            file = urllib.request.urlretrieve("https://github.com/RannerJP/Dog-Image-Classifier/raw/main/models/failModel.h5?download=", ".h5")
            invalid_model = load_model(file[0])
        with self.assertRaises(IncorrectOutputShapeError):
            classifier = ImageClassifier(self.valid_model)
            classifier.set_model(invalid_model)
    
    def test_invalid_image_type(self):
        image = os.path.join('assets', 'Invalid_Image.txt')
        with self.assertRaises(TypeError):
            self.classifier.get_prediction(image)    

    def test_invalid_image_size(self):
        self.classifier.set_model(self.valid_model)
        image = Image.open(os.path.join('assets', 'Valid_Image.jpg'))
        prediction = self.classifier.get_prediction(image)
        self.assertIn(prediction, [-1] + list(range(1,11)))
    
    if __name__ == '__main__':
        unittest.main()