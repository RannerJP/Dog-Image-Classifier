from tensorflow.keras.models import Model
import tkinter as tk
from tkinter import Tk, filedialog
from PIL import Image, ImageTk
import urllib3
from dogclassification.image_classifier import ImageClassifier as image_class
import os
from keras.models import Sequential
from keras.models import load_model
from dogclassification.image_classifier import IncorrectOutputShapeError
class DogClassificationUI:
    def __init__(self, window: Tk, model: Model):
        self.window = window
        self.shown_image = None
        self.classification_text = None
        try:
            self.dog_classifier = image_class(model)
        except TypeError:
            print("inital model invalid, setting to default")
            try:
                self.valid_model = load_model(os.path.join('models', 'DogClassification.h5'))
            except OSError:
                file = urllib3.request.urlretrieve("https://github.com/RannerJP/Dog-Image-Classifier/raw/main/models/DogClassification.h5?download=", ".h5")
                self.valid_model = load_model(file[0])

    def show_UI(self):
        # (For Use When an image is made) self.window.iconbitmap('assets/IMAGE_NAME_HERE.ico')
        frame = tk.Frame(self.window)
        frame.pack(side="top")
        self.window.geometry("260x310")
        self.window.title("Dog Classification")
        upload_model_button = tk.Button(frame, text="Upload Model", command=self.upload_model)
        upload_model_button.pack(side="left")
        upload_image_button = tk.Button(frame, text='Upload image to classify', command=self.upload_file)
        upload_image_button.pack(side="left")

        show_instructions_button = tk.Button(frame, text='Info/Help', command=self.show_instructions)
        show_instructions_button.pack(side="left")
        self.window.mainloop()
    def show_instructions(self):
        instructions = tk.Toplevel(self.window)
        instructions.title("Information")
        instructions_header = "Welcome to the Dog Classifcation program!\n Instructions:"
        instructions_body = "To use this program, Press the \"Upload image to classify\" button."\
                            " Your image gets reformatted to 256x256 in order to make the"\
                            " program to stay a uniform size and since the model accepts image input of 256 x 256"
        information_header = "Information About the Program:"
        information_body = "This program was trained on a CNN with 2 Conv2D layers and a dense layer. Relu activtion was used for the inbetween layers"\
                            " with softmax being used for the final output layer. Each image had at least 150 images in the database, 70% was for training, 20%"\
                                " for validiation, and 10%\ for testing. The program classifies 10 dog breeds, those being: Borzoi, Chihuahua, Dingo, German Shepard"\
                                ", Golden Retriever, Mexican Hairless, Pug, Shih-Tzu, Siberian Husky, and (Standard) Poodle. 72 models with differing parameters "\
                                "were tested with the best one being saved as an h5 file for use in this program. All images were reformatted to 256x256 "\
                                "and came from Standford Dogs Dataset, but only 10 breeds I believed were distinct enough from each other were used. "

        instructions_label = tk.Label(instructions, text = instructions_header, font=("Helvetica", 10, "bold"))
        instructions_label.pack()
        instructions_body_label = tk.Label(instructions, text =instructions_body)
        instructions_body_label.config(wraplength=300)
        instructions_body_label.pack()
        information_header_label = tk.Label(instructions, text=information_header, font=("Helvetica", 10, "bold"))
        information_header_label.pack()
        information_body_label = tk.Label(instructions, text=information_body)
        information_body_label.config(wraplength=300)
        information_body_label.pack()
    def show_prediction(self, image: Image):
        dog_class_predicted = self.dog_classifier.get_prediction(image)
        match dog_class_predicted:
            case 0:
                self.classification_text.configure(text = "Your image is of a: Borzoi")
            case 1:
                self.classification_text.configure(text = "Your image is of a: Chihuahua")
            case 2:
                self.classification_text.configure(text = "Your image is of a: Dingo")
            case 3:
                self.classification_text.configure(text = "Your image is of a: German Shepard")
            case 4:
                self.classification_text.configure(text = "Your image is of a: Golden Retriever")
            case 5:
                self.classification_text.configure(text = "Your image is of a: Mexican Hairless")
            case 6:
                self.classification_text.configure(text = "Your image is of a: Pug")
            case 7:
                self.classification_text.configure(text = "Your image is of a: Shih-Tzu")
            case 8:
                self.classification_text.configure(text = "Your image is of a: Siberian Husky")
            case 9:
                self.classification_text.configure(text = "Your image is of a: Poodle")
            case -1:
                self.classification_text.configure(text = "Your image is not a dog")
            case -2:
                self.classification_text.configure(text = "Model was not of type Sequential. Try loading a new model or restart program")
    def upload_file(self):
        file_types = [('Jpg Files', '*jpg'), ('Png files', '*png')]
        filename = filedialog.askopenfilename(filetypes=file_types)
        image = Image.open(filename)
        image = image.resize((256,256))
        self.show_image(image)

    def set_classification_text(self, image: Image, message = None):
        valid_image = True
        if not isinstance(image, Image.Image):
            valid_image = False
        if self.classification_text is None:
            if message:
                self.classification_text = tk.Label(self.window, text=message) 
                self.classification_text.pack()
            else:
                self.classification_text = tk.Label(self.window, text="Classifying... ") if valid_image else tk.Label(self.window, text="Sorry, that image type is invalid!") 
                self.classification_text.pack()
        else:
            if message:
                self.classification_text.configure(text = message)
            else:
                self.classification_text.configure(text = "Classifying... ") if valid_image else self.classification_text.configure(text = "Sorry, that image type is invalid!")
        return valid_image
    
    def show_image(self, image: Image):
        if isinstance(image, Image.Image):
            image_view = ImageTk.PhotoImage(image)
            if self.shown_image is None:
                self.shown_image = tk.Label(self.window, image=image_view)
                self.shown_image.image = image_view
                self.shown_image.pack()
            else:
                self.shown_image.configure(image = image_view)
                self.shown_image.image = image_view
            self.set_classification_text(image)
            self.window.after(2500, self.show_prediction, image)
        else:
            self.set_classification_text(image)
    def upload_model(self) -> None:
        file_types = [('Keras Files', '*keras'), ('H5 files', '*h5')]
        model_file = filedialog.askopenfilename(filetypes=file_types)
        model = load_model(model_file)
        try:
            self.dog_classifier.set_model(model)
        except IncorrectOutputShapeError as error:
            self.set_classification_text(None, error)


        
        
if __name__ == "__main__":
    """
    resizedImage = tf.image.resize(image, (256,256))
    prediction = model.predict(np.expand_dims(resizedImage/255, 0))
    prediction = prediction.tolist()
    dog_class_predicted = prediction[0].index(max(prediction[0]))
    match dog_class_predicted:
        case 0:
            print("Your image is of a: Borzoi")
        case 1:
            print("Your image is of a: Chihuahua")
        case 2:
            print("Your image is of a: Dingo")
        case 3:
            print("Your image is of a: German Shepard")
        case 4:
            print("Your image is of a: Golden Retriever")
        case 5:
            print("Your image is of a: Mexican Hairless")
        case 6:
            print("Your image is of a: Pug")
        case 7:
            print("Your image is of a: Shih-Tzu")
        case 8:
            print("Your image is of a: Siberian Husky")
        case 9:
            print("Your image is of a: Poodle")
    """
