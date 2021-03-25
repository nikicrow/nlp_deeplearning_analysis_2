# Lets try and do some deep learning models in pycharm
# Importing the final data that we want to analyse

# give credit to https://github.com/rmohashi/emotion-from-tweet/blob/master/notebooks/Train%20Emotion%20Recognition%20Model.ipynb

import pandas as pd
import numpy as np

# download training and test data
train_df = pd.DataFrame(columns=['text', 'labels'])
test_df = pd.DataFrame(columns=['text', 'labels' ])
with open('data/train.txt', 'r', encoding='utf-8') as file:
    for line in file.readlines():
        train_df = train_df.append({'text': line.split(';')[0],
                                    'labels': line.split(';')[1][:-1]
                                    }, ignore_index=True)
with open('data/test.txt', 'r', encoding='utf-8') as file:
    for line in file.readlines():
        test_df = test_df.append({'text': line.split(';')[0],
                                 'labels': line.split(';')[1][:-1]
                                  }, ignore_index=True)

train_data = pd.read_csv('data/training_set.csv')
test_data = pd.read_csv('data/test_set.csv')

# We actually have 2 different sets but i am going to use the second one first because it has more lines
# we will see what happens

### Now we need to turn the text into numbers...
# The basic idea of word embedding is words that occur in similar context tend to be closer to each
# other in vector space. For generating word vectors in Python, modules needed are nltk and gensim.

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import Word2Vec
from pathlib import Path
import re
import nltk
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer


# Preparing the dataset
#all_sentences = nltk.sent_tokenize(all_data_string)
#all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

def clean_data(data):
    # This function takes an array of strings and returns an array of cleaned up strings
    cleaned_data = []
    for row,texts in enumerate(data):
        texts = texts.lower()
        # remove special characters
        texts = texts.replace(r"(http|@)\S+", "")
        texts = texts.replace(r"::", " ")
        texts = texts.replace(r"’", "")
        texts = texts.replace(r",", " ")
        texts = texts.replace(r"[^a-z\':_]", " ")
        # remove repetition
        #pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        #texts = texts.replace(pattern, r"\1")
        # Transform short negation form
        texts = texts.replace(r"(can't|cannot)", 'can not')
        texts = texts.replace(r"n't", ' not')
        # Remove stop words
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.remove('not')
        stopwords.remove('nor')
        stopwords.remove('no')
        cleaned_line = ''
        for word in texts.split(" "):
            if word not in stopwords:
                cleaned_line = cleaned_line + " " + word
        cleaned_data.append(cleaned_line)
    return cleaned_data


train_data['Text'] = clean_data(train_data['Text'])
test_data['Text'] = clean_data(test_data['Text'])

# Now that the data is cleaned up we need to tokenize it
num_words = 10000

tokenizer = Tokenizer(num_words=num_words, lower=True)
tokenizer.fit_on_texts(train_data.Text)

file_to_save = Path('../nlp_deeplearning_analysis/models/tokenizer.pickle').resolve()
with file_to_save.open('wb') as file:
    pickle.dump(tokenizer, file)

# The data is tokenized - now lets try putting it in the model

from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model

# define model params
input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes = len(train_data.Emotion.unique())
embedding_dim = 500
input_length = 100
lstm_units = 128
lstm_dropout = 0.1
recurrent_dropout = 0.1
spatial_dropout=0.2
filters=64
kernel_size=3

# create model
input_layer = Input(shape=(input_length,))
output_layer = Embedding(
  input_dim=input_dim,
  output_dim=embedding_dim,
  input_shape=(input_length,)
)(input_layer)

output_layer = SpatialDropout1D(spatial_dropout)(output_layer)

output_layer = Bidirectional(
LSTM(lstm_units, return_sequences=True,
     dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
)(output_layer)
output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                    kernel_initializer='glorot_uniform')(output_layer)

avg_pool = GlobalAveragePooling1D()(output_layer)
max_pool = GlobalMaxPooling1D()(output_layer)
output_layer = concatenate([avg_pool, max_pool])

output_layer = Dense(num_classes, activation='softmax')(output_layer)

model = Model(input_layer, output_layer)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Now lets prep the model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

# turn into arrays of words not strings
train_sequences = [text.split() for text in train_data.Text]
validation_sequences = [text.split() for text in test_data.Text]
# turn into array of integers
list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)
# pad out the arrays
x_train = pad_sequences(list_tokenized_train, maxlen=input_length)
x_validation = pad_sequences(list_tokenized_validation, maxlen=input_length)
# encode the data
encoder = LabelBinarizer()
encoder.fit(train_data.Emotion.unique())

encoder_path = Path('../nlp_deeplearning_analysis/models/emotion_recognition', 'encoder.pickle')
with encoder_path.open('wb') as file:
    pickle.dump(encoder, file)

y_train = encoder.transform(train_data.Emotion)
y_validation = encoder.transform(test_data.Emotion)

# Ready to train the model now

batch_size = 128
epochs = 3

history = model.fit(
    x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_validation, y_validation)
)

# save the model
#model_file = Path('../models/emotion_recognition/model_weights.h5').resolve()
#model.save_weights(model_file.as_posix())

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(ymin=0)
plt.show()

# how good are you my model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#predictions = model.predict(x_validation)
#print(confusion_matrix(y_validation,predictions))
#print(classification_report(y_validation,predictions))
#print(accuracy_score(y_validation, predictions))

# The model has been trained and has decent accuracy. We need to now take data that we have and
# use the model to predict the emotions

survey_data = pd.read_excel('data/raw_data.xlsx')

columns_with_open_responses = ['Q02', 'Q04', 'Q06', 'Q08', 'Q10', 'Q12', 'Q14', 'Q16', 'Q18']
raw_data = survey_data[columns_with_open_responses]

current_column = 'Q06'###UPDATE HERE
current_data = raw_data[current_column]
current_data = current_data.dropna()
current_data = clean_data(current_data)

# turn the data into a vector using the tokenizer from the model
tokenizer.fit_on_texts(current_data)
list_tokenized_data = tokenizer.texts_to_sequences(current_data)
x_padded = pad_sequences(list_tokenized_data, maxlen=input_length)

predicted = model.predict(x_padded)

emotions_dictionary = { '1' : 'anger',
                        '2' : 'fear',
                        '3' : 'joy',
                        '4' : 'neutral',
                        '5' : 'sadness'}

averages = predicted.mean(axis=0)
emotions_df = pd.DataFrame(columns=['Emotion','Average'])

for i,number in enumerate(averages):
    emotions_df = emotions_df.append({ 'Emotion' : emotions_dictionary[str(i+1)],
                                       'Average' : number},
                                     ignore_index=True)
    print(emotions_dictionary[str(i+1)]," = ",number)



# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
###UPDATE HERE on the Q number
df = pd.DataFrame.from_records([{
    'Emotion': 'Q06',
    'Anger': emotions_df['Average'][0]*100,
    'Fear': emotions_df['Average'][1]*100,
    'Joy': emotions_df['Average'][2]*100,
    'Neutral': emotions_df['Average'][3]*100,
    'Sadness': emotions_df['Average'][4]*100
}])

def spider_chart(df):
    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = df.loc[0].drop('Emotion').values.flatten().tolist()
    values += values[:1]
    values

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    plt.ylim(0, 40)
    plt.title('Feedback Emotions from Q4') ###UPDATE HERE

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    # Show the graph
    plt.show()
    return


spider_chart(df)

'''
from gensim.models import Word2Vec

word2vec = Word2Vec(all_words, min_count=2)

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

def words_to_numbers(raw_data):
    return_data = []
    # iterate through each sentence in the file
    for line in range(len(raw_data)):
        for i in sent_tokenize(raw_data[line]):
            temp = []

            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())

            return_data.append(temp)
    # Create CBOW model
    model1 = gensim.models.Word2Vec(return_data, min_count=1,
                                    size=100, window=5)
    return model1, return_data

# insightful - https://stackabuse.com/implementing-word2vec-with-gensim-library-in-python/

model1, X_train = words_to_numbers(train_data['Text'])
model2, X_test = words_to_numbers(test_data['Text'])



## Let's try a Decision Tree first
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_decision_tree(X_train, y_train, X_test, y_test):
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)

    print(classification_report(y_test, predictions))
    return "Completed"

#RANDOM_FOREST = train_random_forest(X_train, y_train, X_test, y_test)
'''