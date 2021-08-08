import numpy as np
import pandas as pd
import re
import string

str_punc = string.punctuation.replace(',', '').replace("'",'')
def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text    

# train = pd.read_csv('../input/goemotion/train.tsv', sep='\t')
# test = pd.read_csv('../input/goemotion/test.tsv', sep='\t')
# dataset = pd.concat((train, test))
# dataset = pd.concat((pd.read_csv('../input/bestemo/data_train.csv'), pd.read_csv('../input/bestemo/data_test.csv')))

# goEmotion = pd.read_csv('../input/emotion/dataset_classified.csv')
# isear = pd.read_csv('../input/emotion/isear_dataset.csv')
# twitter = pd.read_csv('../input/emotion/tweet_emotions.csv')
# isear.rename(columns={'Emotion':'sentiment', 'Text':'content'}, inplace=True)
# twitter.drop('tweet_id', axis=1, inplace=True)
# isear.head()
# twitter_neutral = twitter[twitter['sentiment'].isin(['neutral', 'happiness', 'sadness'])]
# isear = pd.concat((isear, twitter_neutral))
# isear.loc[isear['sentiment'] == 'happiness']['sentiment'] = 'joy'
# # dataset = pd.read_csv('../input/emolarge/Emotion_Large.csv')
# dataset = isear
# dataset.dropna(inplace=True)

# dataset = pd.read_csv('../input/emotionpure/pure_dataset.csv')
# neutral_sample = dataset.loc[dataset['sentiment'] == 'neutral'].sample(5000)
# anger_sample = dataset.loc[dataset['sentiment'] == 'anger']
# joy_sample = dataset.loc[dataset['sentiment'] == 'joy'].sample(5000)
# sorrow_sample = dataset.loc[dataset['sentiment'] == 'sorrow'].sample(5000)
# fear_sample = dataset.loc[dataset['sentiment'] == 'fear'].sample(5000)
# # dataset = dataset.loc[~dataset['sentiment'].isin(['neutral', 'fear'])]
# dataset = pd.concat((neutral_sample, fear_sample, anger_sample, joy_sample, sorrow_sample))
dataset = pd.read_csv('../input/emotion/isear_dataset.csv')
print(dataset['Emotion'].value_counts())
# dataset = dataset[dataset['sentiment'].isin(['neutral', 'joy', 'sadness'])]

def getDatasetByClassName(dataset, classes, column='Emotion'):
    # Function to return a sub-dataset by a condition on the column values
    return dataset.loc[dataset[column].isin(classes)]

def combineDatasetsByClassName(datasets, classes):
    # Function to combine datasets on the common column value
    combined_dataset = getDatasetByClassName(dataset=datasets[0], classes=classes)
    for i in range(1, len(datasets)):
        dataset_single_class = getDatasetByClassName(dataset=datasets[i], classes=classes)
        combined_dataset = pd.concat((combined_dataset, dataset_single_class))
    return combined_dataset


X = dataset['Text'].apply(clean)
y = dataset['Emotion']

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)
# y = y[:, 0].astype(int)
# np.unique(y, return_counts=True)
# y = y[:, 1]

from sklearn.model_selection import train_test_split
text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)
sequences_train = tokenizer.texts_to_sequences(text_train)
sequences_test = tokenizer.texts_to_sequences(text_test)
X_train = pad_sequences(sequences_train, maxlen=48, truncating='pre')
X_test = pad_sequences(sequences_test, maxlen=48, truncating='pre')
vocabSize = len(tokenizer.index_word) + 1
vocabSize




# def create_embedding_matrix(filepath, word_index, embedding_dim):
#     vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
#     embedding_matrix = np.zeros((vocab_size, embedding_dim))
#     with open(filepath, encoding="utf8") as f:
#         for line in f:
#             word, *vector = line.split()
#             if word in word_index:
#                 idx = word_index[word] 
#                 embedding_matrix[idx] = np.array(
#                     vector, dtype=np.float32)[:embedding_dim]
#     return embedding_matrix

# embed_mtx = create_embedding_matrix('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt', tokenizer.index_word, 200)
# embed_mtx.shape

path_to_glove_file = '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'
num_tokens = vocabSize
embedding_dim = 200
hits = 0
misses = 0
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))
embed_mtx = embedding_matrix

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding, Bidirectional
from keras.optimizers import SGD
opt = SGD(lr=0.001)

model = Sequential()
model.add(Embedding(vocabSize, 200, input_length=X_train.shape[1], weights=[embed_mtx], trainable=False))
model.add(Bidirectional(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.2,recurrent_dropout=0.2, return_sequences=False)))

# model.add(Bidirectional(LSTM(64, dropout=0.2,recurrent_dropout=0.2)))
# model.add(Bidirectional(LSTM(64, dropout=0.2,recurrent_dropout=0.2)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1, batch_size=128, epochs=1)

mysentence = "Nothing"
# mysentence = "I am never happy"
mysentence = clean(mysentence)
print(mysentence)
mysentence = tokenizer.texts_to_sequences([mysentence])
print(mysentence)
mysentence = pad_sequences(mysentence, maxlen=48, truncating='pre')
emotion = le.inverse_transform(model.predict_classes(mysentence))[0]
proba =  model.predict_proba(mysentence)[0]
print(emotion, proba)

model.evaluate(X_test, y_test, verbose=1)

import pickle
with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
    
with open('labelEncoder.pickle', 'wb') as f:
    pickle.dump(le, f)
    
    
model.save('emotionRecognition75.h5')

# %matplotlib inline
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()

# plt.plot(epochs, loss, 'r', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

# print(history.history)