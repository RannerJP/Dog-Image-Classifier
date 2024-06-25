import numpy as np
class ImageClassifier:
    """
    The ImageClassifier Class handles any predictions made with the model

    Attributes:
        attribute1 (int): Description of attribute1.
        attribute2 (str): Description of attribute2.
    """
    def __init__(self, model):
        self.model = model

    def get_prediction(self, np_image):
        prediction = self.model.predict(np.expand_dims(np_image/255, 0))
        prediction = prediction.tolist()
        max_prediction_percent = max(prediction[0])
        dog_class_predicted = prediction[0].index(max_prediction_percent)
        return dog_class_predicted
    
    def set_model(self, model_to_set):
        self.model = model_to_set