import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageTk

# Load the tokenizer (assuming you have saved it as tokenizer.pickle)
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('best_model.h5')

# Define the maximum length of sequences
max_len = 50

# Function to preprocess the input text and make a prediction
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, padding='post', maxlen=max_len)
    return padded_sequence

def predict_sentiment():
    text = text_box.get("1.0", tk.END).strip()
    if text:
        processed_text = preprocess_text(text)
        prediction = model.predict(processed_text)
        sentiment = np.argmax(prediction, axis=1)[0]

        sentiment_classes = ['Negative', 'Neutral', 'Positive']
        result = sentiment_classes[sentiment]

        messagebox.showinfo("Sentiment Analysis Result", f"The sentiment is: {result}")
    else:
        messagebox.showwarning("Input Error", "Please enter some text to analyze.")

# Create the main application window
root = tk.Tk()
root.title("Sentiment Analysis")

# Add an image
img = Image.open("new.jpg")  # Replace with the path to your image
img = img.resize((300, 200), Image.LANCZOS)  # Resize the image to fit the window
photo = ImageTk.PhotoImage(img)
image_label = tk.Label(root, image=photo)
image_label.pack(pady=10)

# Create and place the text box
text_box = tk.Text(root, height=10, width=50)
text_box.pack(pady=10)

# Create and place the analyze button
analyze_button = tk.Button(root, text="Check Sentiment of text", command=predict_sentiment)
analyze_button.pack(pady=10)

# Run the application
root.mainloop()
