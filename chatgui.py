# Tắt GPU nếu có
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# Creating GUI with tkinter
import tkinter as tk
from tkinter import *
def send(event=None):
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n', ('user',))
        ChatLog.config(foreground="#00254d", font=("Arial", 12))

        res = chatbot_response(msg)
        ChatLog.insert(tk.END, "Bot: " + res + '\n\n', ('bot',))

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

base = tk.Tk()
base.title("Lipstick chatbot")
base.geometry("500x600")
base.resizable(width=tk.FALSE, height=tk.FALSE)
ChatLog = Text(base, bd=0, bg="white", height="8", width=100, font="Arial", wrap="word")
ChatLog.config(state=tk.DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="dot")
ChatLog['yscrollcommand'] = scrollbar.set
EntryBox = tk.Text(base, bd=0, bg="white", width=29, height="5", font="Arial")
SendButton = tk.Button(base, font=("Arial", 13, 'bold'), text="Send", width="10", height=5,
                        bd=0, bg="#003366", activebackground="#195794", fg='#ffffff', command=send)
# also send when press Enter
EntryBox.bind("<Return>", send)

# Place all components on the screen
scrollbar.place(x=476, y=6, height=586)
ChatLog.place(x=6, y=6, height=510, width=470)
EntryBox.place(x=125, y=525, height=60, width=344)
SendButton.place(x=6, y=525, height=60)

base.mainloop()