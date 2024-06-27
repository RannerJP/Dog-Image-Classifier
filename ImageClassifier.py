import numpy as np
from tensorflow.keras.models import Model
from PIL import Image
class ImageClassifier:
    """
    The ImageClassifier Class handles any predictions made with the model
    """
    def __init__(self, model: Model):
        self.model = model

    def get_prediction(self, image: Image):
        np_image = np.array(image)
        prediction = self.model.predict(np.expand_dims(np_image/255, 0))
        prediction = prediction.tolist()
        max_prediction_percent = max(prediction[0])
        dog_class_predicted = prediction[0].index(max_prediction_percent)
        return dog_class_predicted
    
    def set_model(self, model_to_set: Model):
        self.model = model_to_set