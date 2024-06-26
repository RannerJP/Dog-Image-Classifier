import numpy as np
from tensorflow.keras.models import Model
from keras.models import Sequential
from PIL import Image
class IncorrectOutputShapeError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ImageClassifier:
    """
    The ImageClassifier Class handles any predictions made with the model
    """
    def __init__(self, model: Sequential):
        if not isinstance(model, Sequential):
            raise TypeError("The model should be a Keras Sequential model.")
        if model.layers[-1].units != 10:
            raise IncorrectOutputShapeError(f"Model has incorrect number of outputs. Expected 10 got {model.layers[-1].units}")
        self.model = model

    def get_prediction(self, image: Image):
        if not isinstance(self.model, Sequential):
            return -2
        if not isinstance(image, Image.Image):
            raise TypeError("user inputted photo is not of type Image.Image")
        if image.size != (256, 256):
            image = image.resize((256, 256))
        np_image = np.array(image)
        prediction = self.model.predict(np.expand_dims(np_image/255, 0))
        prediction = prediction.tolist()
        max_prediction_percent = max(prediction[0])
        dog_class_predicted = prediction[0].index(max_prediction_percent)
        return dog_class_predicted
    
    def set_model(self, model_to_set: Model):
        if not isinstance(model_to_set, Sequential):
            raise TypeError("The model should be a Keras Sequential model.")
        if model_to_set.layers[-1].units != 10:
            raise IncorrectOutputShapeError(f"Model has incorrect number of outputs. Expected 10 got {model_to_set.layers[-1].units}")
        self.model = model_to_set