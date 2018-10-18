import pandas as pd
import numpy as np
import os
from os.path import join
import re
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout, recurrent
from keras.callbacks import Callback
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, rand
from numpy import zeros as np_zeros

class Configuration(object):
    """Dump stuff here"""

CONFIG = Configuration()
CONFIG.input_layers = 1
CONFIG.output_layers = 1
CONFIG.amount_of_dropout = 0.2
CONFIG.hidden_size = 200
CONFIG.initialization = "he_normal" # : Gaussian initialization scaled by fan-in (He et al., 2014)
CONFIG.number_of_chars = 100
CONFIG.max_input_len = 20
CONFIG.inverted = True

# parameters for the training:
CONFIG.batch_size = 100 # As the model changes in size, play with the batch size to best fit the process in memory
CONFIG.epochs = 10 # due to mini-epochs.
CONFIG.steps_per_epoch = 1000 # This is a mini-epoch. Using News 2013 an epoch would need to be ~60K.
CONFIG.validation_steps = 10
CONFIG.number_of_iterations = 5

DATA_FILES_PATH ="./Books_English"
DATA_FILES_FULL_PATH = os.path.expanduser(DATA_FILES_PATH)
NEWS_FILE_NAME_ENGLISH = "news.2007.en.shuffled"
NEWS_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, NEWS_FILE_NAME_ENGLISH)
NEWS_FILE_NAME_CLEAN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.clean")
NEWS_FILE_NAME_SPLIT = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.split")

# Some cleanup:
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE) # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]')
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)), re.UNICODE)

PADDING = "☕"
MIN_INPUT_LEN = 5
AMOUNT_OF_NOISE =  0.2 / CONFIG.max_input_len
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")

#Clean up the text
def clean_text(text):
    result = NORMALIZE_WHITESPACE_REGEX.sub(' ', text.strip())
    result = RE_DASH_FILTER.sub('-', result)
    result = RE_APOSTROPHE_FILTER.sub("'", result)
    result = RE_LEFT_PARENTH_FILTER.sub("(", result)
    result = RE_RIGHT_PARENTH_FILTER.sub(")", result)
    result = RE_BASIC_CLEANER.sub('', result)
    return result

def preprocesses_data_clean():
    with open(NEWS_FILE_NAME_CLEAN, "wb") as clean_data:
        for line in open(NEWS_FILE_NAME):
            decoded_line = line
            cleaned_line = clean_text(decoded_line)
            encoded_line = cleaned_line.encode("utf-8")
            clean_data.write(encoded_line + b"\n")
            
#Seperate the text into lines
def preprocesses_split_lines():
    answers = set()
    with open(NEWS_FILE_NAME_SPLIT, "wb") as output_file:
        for _line in open(NEWS_FILE_NAME_CLEAN):
            line = _line
            while len(line) > MIN_INPUT_LEN:
                if len(line) <= CONFIG.max_input_len:
                    answer = line
                    line = ""
                else:
                    space_location = line.rfind(" ", MIN_INPUT_LEN, CONFIG.max_input_len - 1)
                    if space_location > -1:
                        answer = line[:space_location]
                        line = line[len(answer) + 1:]
                    else:
                        space_location = line.rfind(" ") # no limits this time
                        if space_location == -1:
                            break # we are done with this line
                        else:
                            line = line[space_location + 1:]
                            continue
                answers.add(answer)
                output_file.write(answer.encode('utf-8') + b"\n")
                

#Add error to the data to creat misspelling
def add_noise_to_string(a_string, amount_of_noise):
    if rand() < amount_of_noise * len(a_string):
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
    if rand() < amount_of_noise * len(a_string):
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
    if len(a_string) < CONFIG.max_input_len and rand() < amount_of_noise * len(a_string):
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
    if rand() < amount_of_noise * len(a_string):
        random_char_position = random_randint(len(a_string) - 1)
        a_string = (a_string[:random_char_position] + a_string[random_char_position + 1] + a_string[random_char_position] +
                    a_string[random_char_position + 2:])
    return a_string

#Unify the dimetion of the data
def generate_question(answer):
    question = add_noise_to_string(answer, AMOUNT_OF_NOISE)
    question += PADDING * (CONFIG.max_input_len - len(question))
    answer += PADDING * (CONFIG.max_input_len - len(answer))
    return question, answer

#Generate data
def generate_news_data():
    answers = open(NEWS_FILE_NAME_SPLIT).read().split("\n")
    questions = []
    random_shuffle(answers)
    for answer_index, answer in enumerate(answers):
        question, answer = generate_question(answer)
        answers[answer_index] = answer
        assert len(answer) == CONFIG.max_input_len
        questions.append(question)

    return questions, answers

#encode and decode a character set
class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    @property
    def size(self):
        return len(self.chars)

    def encode(self, C, maxlen):
        X = np_zeros((maxlen, len(self.chars)), dtype=np.bool)
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X if x)

#Vectorize the data and make the format which we can feed to to network
def _vectorize(questions, answers, ctable):
    len_of_questions = len(questions)
    X = np_zeros((len_of_questions, CONFIG.max_input_len, ctable.size), dtype=np.bool)
    #print(len(questions))
    print("888888888888888888888888",len(questions))
    for i in range(len(questions)):
        sentence = questions.pop()
        print("999999999999999999999999",len(questions))
        print("777777777777777777777777",len(answers))
        for j, c in enumerate(sentence):
            try:
                X[i, j, ctable.char_indices[c]] = 1
                #print("I AM HERE",i,j,X[i, j, ctable.char_indices[c]])
            except KeyError:
                pass
    y = np_zeros((len_of_questions, CONFIG.max_input_len, ctable.size), dtype=np.bool)
    for i in range(len(answers)):
        sentence = answers.pop()
        for j, c in enumerate(sentence):
            try:
                y[i, j, ctable.char_indices[c]] = 1
            except KeyError:
                pass
    return X, y

def vectorize(questions, answers, chars,load_all_files_as_test=False):
    save_characters(chars)
    ctable = CharacterTable(chars)
    print(len(questions))
    print(len(answers))
    print(questions[0])
    print(answers[0])
    X, y = _vectorize(questions, answers, ctable)
    print("++++++++++++++++++++++=",ctable.decode(X[len(X)-1,:]))
    print("++++++++++++++++++++++=",ctable.decode(y[len(y)-1,:]),len(y))
    split_at = int(len(X) - len(X) / 10)
    if not load_all_files_as_test:
        X_train, X_test= train_test_split(X, test_size = 0.1, random_state = 2)
        y_train, y_test= train_test_split(y, test_size = 0.1, random_state = 2)
    
        X_train,X_val= train_test_split(X_train, test_size = 0.1, random_state = 2)
        y_train,y_val= train_test_split(y_train, test_size = 0.1, random_state = 2)
    else:
        X_train=X
        y_train=y
        X_test=[]
        X_val=[]
        y_test=[]
        y_val=[]
    return X_train, X_val, X_test, y_train, y_val, y_test, CONFIG.max_input_len, ctable


#Building the Model
def generate_model(output_len, chars=None):
    model = Sequential()
    for layer_number in range(CONFIG.input_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, input_shape=(None, len(chars)),return_sequences=layer_number + 1 < CONFIG.input_layers))
        model.add(Dropout(CONFIG.amount_of_dropout))
    model.add(RepeatVector(output_len))
    for _ in range(CONFIG.output_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, return_sequences=True, kernel_initializer=CONFIG.initialization))
        model.add(Dropout(CONFIG.amount_of_dropout))
    model.add(TimeDistributed(Dense(len(chars), kernel_initializer=CONFIG.initialization)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#train the model for all iteration steps
def iterate_training(model, X_train, y_train, X_val, y_val, ctable):
    for iteration in range(1, CONFIG.number_of_iterations):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=CONFIG.batch_size, epochs=CONFIG.epochs)
        model.save('my_model.h5')
        model.save_weights('my_model_weights.h5')

def train_model():
    questions, answers = generate_news_data()
    chars_answer = set.union(*(set(answer) for answer in answers))
    chars_question = set.union(*(set(question) for question in questions))
    chars = list(set.union(chars_answer, chars_question))
    X_train, X_val, X_test, y_train, y_val, y_test, y_maxlen, ctable = vectorize(questions, answers, chars)
    print ("y_maxlen, chars", y_maxlen, "".join(chars))
    model = generate_model(y_maxlen, chars)
    iterate_training(model, X_train, y_train, X_val, y_val, ctable)
    validate_model_using_test(X_test,y_test)
    
def validate_model_using_test(xtest,ytest):
    model = load_model('my_model.h5')
    test_loss, test_acc =model.evaluate(xtest, ytest)
    print('Accuracy for the test data set:', test_acc)

def save_characters(chars):
    with open('char_list.txt', 'w') as cl:
        for item in chars:
            cl.write("%s\n" % item)

def load_characters():
    with open('char_list.txt') as cl:
        chars = cl.read().splitlines()
    return chars

def evaluate_trained_model(X_given):
    chars=load_characters()
    model = load_model('my_model.h5')
    answers = []
    questions = []
    for answer_index, answer in enumerate(X_given):
        question, answer = generate_question(answer)
        answers.append(answer)
#        print("A",answers[answer_index])
#        print("Q",questions[answer_index])
        assert len(answer) == CONFIG.max_input_len
        questions.append(question)
#    questions[0]="HEyu"
    print(questions)
    print(answers)
    length_x=len(x)
    X_train, X_val, X_test, y_train, y_val, y_test, y_maxlen, ctable = vectorize(questions, answers, chars,load_all_files_as_test=True)
    print("-------------------",ctable.decode(y_train[length_x-1,:]))
    #print(y_train[0])
    #print(ctable.decode(y_train[0]))
#    pred=model.predict_classes(y_train[length_x-1,:], verbose=0)
#    ctable.decode(pred.all(), calc_argmax=False)
    #print(pred.shape)
    #for i in range(20):
        #print(ctable.decode(pred[i-1], calc_argmax=False))

if __name__ == '__main__':
    preprocesses_data_clean()
    preprocesses_split_lines()
    
#    train_model()
    x=[]
    x.append("Love")
    evaluate_trained_model(x)