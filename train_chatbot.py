import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '/', '\\', '']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

# training set, create bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words arr with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Create model 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neuron
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Thêm lớp Dense với 128 neurons, hàm kích hoạt là ReLU
model.add(Dropout(0.5))  # Thêm lớp Dropout để tránh overfitting
model.add(Dense(64, activation='relu'))  # Thêm lớp Dense với 64 neurons, hàm kích hoạt là ReLU
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  # Lớp đầu ra với số neurons bằng số lượng intents, hàm kích hoạt là Softmax


sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # Tạo optimizer SGD với các tham số cụ thể
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Biên soạn mô hình với categorical crossentropy loss và SGD optimizer

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)# Huấn luyện mô hình với 200 epochs và batch_size là 5
model.save('chatbot_model.h5', hist)

print("model created")
