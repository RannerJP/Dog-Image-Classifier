from keras.models import load_model
import os
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
from PIL import Image, ImageTk
shown_image = None
classification_text = None
def show_instructions():
    instructions = tk.Toplevel(window)
    instructions.title("Information")
    instructions_header = "Welcome to the Dog classifcation program!\n Instructions:"
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
def show_prediction(np_image):
    prediction = model.predict(np.expand_dims(np_image/255, 0))
    prediction = prediction.tolist()
    global classification_text
    max_prediction_percent = max(prediction[0])
    dog_class_predicted = prediction[0].index(max_prediction_percent)
    match dog_class_predicted:
        case 0:
            classification_text.configure(text = "Your image is of a: Borzoi")
        case 1:
            classification_text.configure(text = "Your image is of a: Chihuahua")
        case 2:
            classification_text.configure(text = "Your image is of a: Dingo")
        case 3:
            classification_text.configure(text = "Your image is of a: German Shepard")
        case 4:
            classification_text.configure(text = "Your image is of a: Golden Retriever")
        case 5:
            classification_text.configure(text = "Your image is of a: Mexican Hairless")
        case 6:
            classification_text.configure(text = "Your image is of a: Pug")
        case 7:
            classification_text.configure(text = "Your image is of a: Shih-Tzu")
        case 8:
            classification_text.configure(text = "Your image is of a: Siberian Husky")
        case 9:
            classification_text.configure(text = "Your image is of a: Poodle")
        case -1:
            classification_text.configure(text = "Your image is not a dog")
def upload_file():
    file_types = [('Jpg Files', '*jpg'), ('Png files', '*png')]
    filename = filedialog.askopenfilename(filetypes=file_types)
    image = Image.open(filename)
    image = image.resize((256,256))
    image_view = ImageTk.PhotoImage(image)
    global shown_image
    global classification_text
    if shown_image is None:
        shown_image = tk.Label(window, image=image_view)
        shown_image.image = image_view
        shown_image.pack()
    else:
        shown_image.configure(image = image_view)
        shown_image.image = image_view
    np_image = np.array(image)
    if classification_text is None:
        classification_text = tk.Label(window, text="Classifying... ")
        classification_text.pack()
    else:
        classification_text.configure(text = "Classifying... ")
    window.after(3000, show_prediction, np_image)
    

    
    
            
model = load_model(os.path.join('models', 'DogClassification_text.h5'))
window = tk.Tk()
frame = tk.Frame(window)
frame.pack(side="top")
window.geometry("260x310")
window.title("Dog Classification_texts")
button1 = tk.Button(frame, text='Upload image to classify', command=upload_file)
button1.pack(side="left")

button2 = tk.Button(frame, text='Info/Help', command=show_instructions)
button2.pack(side="left")
window.mainloop()
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
