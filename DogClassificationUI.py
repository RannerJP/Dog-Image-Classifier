import os
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import Tk, filedialog
from PIL import Image, ImageTk
from ImageClassifier import ImageClassifier as image_class
class DogClassificationUI:
    def __init__(self, window: Tk, model: Model):
        self.window = window
        self.shown_image = None
        self.classification_text = None
        self.dog_classifier = image_class(model)
    def show_UI(self):
        frame = tk.Frame(window)
        frame.pack(side="top")
        window.geometry("260x310")
        window.title("Dog Classification")
        button1 = tk.Button(frame, text='Upload image to classify', command=self.upload_file)
        button1.pack(side="left")

        button2 = tk.Button(frame, text='Info/Help', command=self.show_instructions)
        button2.pack(side="left")
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
    def upload_file(self):
        file_types = [('Jpg Files', '*jpg'), ('Png files', '*png')]
        filename = filedialog.askopenfilename(filetypes=file_types)
        image = Image.open(filename)
        image = image.resize((256,256))
        self.show_image(image)

    def show_image(self, image: Image):
        image_view = ImageTk.PhotoImage(image)
        if self.shown_image is None:
            self.shown_image = tk.Label(self.window, image=image_view)
            self.shown_image.image = image_view
            self.shown_image.pack()
        else:
            self.shown_image.configure(image = image_view)
            self.shown_image.image = image_view
        if self.classification_text is None:
            self.classification_text = tk.Label(self.window, text="Classifying... ")
            self.classification_text.pack()
        else:
            self.classification_text.configure(text = "Classifying... ")
        self.window.after(3000, self.show_prediction, image)
        

        
        
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
