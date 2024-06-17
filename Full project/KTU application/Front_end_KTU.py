#Importing all needed libraries for the machine learning model
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np 
import re 
from nltk.tokenize import word_tokenize
#create the model and vectorizer objects to be used for the code
model = joblib.load('KTU complete setup\saved_models\KTU_model.joblib')
vectorizer = joblib.load('KTU complete setup\saved_models\KTU_vectorizer.joblib')

#to remove whitespaces , emojies and other non text and numerical characters
def clean_text(text):
    final_text = text.strip().lower()
    emoji_pattern = r'[^a-zA-Z0-9\s]+'
    final_text = re.sub(emoji_pattern , '', final_text)
    return final_text

#The function to tokenize the data for further NLP tasks in preparation for making predictions
def tokenize_text(text):
    tokenized_text = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = []
    for token in tokenized_text:
        if token in tokenized_text: 
            if token not in stop_words:
                tokens.append(token)
    return tokens

#Final text to be fed into the vectorizer
def generate_pure_text(tokens):
    final_text = ''
    for token in tokens: 
        final_text += token
        final_text += ' '
    return final_text
#This is the data pipeline 
def Process_data(text):
    #First we clean the text of any characters that may cause errors
    text_data = clean_text(text)
    #Tokenizing the data so we can only extract the most important data from the text
    tokens = tokenize_text(text_data)
    #Regenerating text with only the most relevant words in it
    purified_text = generate_pure_text(tokens)
    return purified_text

def make_predictions(text):
    #Feeding the text through the pipeline first 
    text_data = Process_data(text)
    #Vectorizing the text
    data = vectorizer.transform([text_data])
    #Making the predictions
    prediction = model.predict(data)
    #returning the predictions
    return prediction

#functions for the pipeline to clean and prepare the needed data
"""
All Code below this point is for the backend of application
"""

"""
All code below corresponds to the UI of the application and its design.
"""
# Importing library for UI
import tkinter as tk 
from tkinter import messagebox 
#Function to load text from the textbox into the label 
def predict_text():
    text = text_box.get("1.0", tk.END).strip()
    username = UserID.get("1.0", tk.END).strip()
    if text: 
        answer = make_predictions(text)
        result = (answer[0].strip().lower() == username.lower())
        if result == True : 
            label.config(text = "Access Granted")
        else: 
            label.config(text = "Access Denied")
        print(answer)
    else: 
        messagebox.showwarning("Input Error", "Please enter some text to make predictions")

#Function to clear the text in the label
def clear_text():
    text_box.delete("1.0",tk.END)
    UserID.delete("1.0", tk.END)
    label.config(text = "")

#Create the main window
root = tk.Tk()

# Set the title of the window
root.title("KTU Prototype")

#Set the size of the window
root.geometry("400x300")

#User name label to make it easier to understand
userlabel = tk.Label(root , text ="UserID")
userlabel.pack(pady= 10)
#Username Box
UserID = tk.Text(root , height = 1 , width= 40)
UserID.pack(pady = 10)
#Data box label
datalabel = tk.Label(root , text = "Enter data in here")
datalabel.pack(pady = 10)
#Create a textbox widget
text_box = tk.Text(root, height= 5, width=40)
text_box.pack(pady = 10 )


#Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady = 10)

#Create "Load Text" button
load_button = tk.Button(button_frame, text="Predict", command=predict_text)
load_button.pack(side=tk.LEFT,padx = 5)

#Create "Clear Text" button 
clear_button = tk.Button(button_frame, text="Clear Text", command= clear_text)
clear_button.pack(side = tk.LEFT, padx=5)

#Create a label widget
label = tk.Label(root, text="")
label.pack(pady=10)

#Run the application
root.mainloop()