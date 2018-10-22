import pandas as pd
from numpy import array
from pickle import dump
import textwrap
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense,LSTM, Dropout, RepeatVector,Activation, TimeDistributed
# load doc into memory
DATA_FILENAME='news.txt'
CLEANDATA_FILENAME='news_clean.txt'
MAX_LEN_LINE=100
VALID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .0123456789,()-"
VOCAB_SIZE=len(VALID_CHARS)
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]
#Read the text data to be used for training
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

#Save a text file
def save_doc(lines, filename):
	#data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(lines)
	file.close()
#Clean the data according to the provided valid characters
def clean_data(file_name,file_name_clean):
    # load text
    raw_text = load_doc(file_name)
    raw_text=' '.join([line.strip() for line in raw_text.strip().splitlines()])
    raw_text = ''.join([char for char in raw_text if char in VALID_CHARS])
    raw_text=raw_text.replace('. ','.\n')
    save_doc(raw_text,file_name_clean)

#Load the clean data from the file
def load_clean_data(file_name):
    with open(file_name) as cf:
        clean_lines = cf.readlines()
    clean_lines = [x.strip() for x in clean_lines] 
    return clean_lines
#Truncate the sentences to the proper size of characters provided by the user
def extract_proper_sentences(text_data):
    sentences=[]
    for line in range(len(text_data)):
        truncated_line=text_data[line]
        truncated_line=truncated_line[0:MAX_LEN_LINE]
        sentences.append(truncated_line)
    return sentences

#Build a dictionary of character to be used for encoding and decoding the sentences
def build_dictionary_of_characters(chars):
# Create a dictionary to convert the vocabulary (characters) to integers
    vocab_int = {}
    count = 0
    for character in chars:
        vocab_int[character] = count
        count += 1
#   CHARACTER TO USE FOR PADDING
    vocab_int['$'] = count


    # Create another dictionary to convert integers to their respective characters
    int_vocab = {}
    for character, value in vocab_int.items():
        int_vocab[value] = character
    return vocab_int,int_vocab

#Convert the sentences from character to integer
def convert_sentences_to_integer(sentences,vocab_int):
    int_sentences = []
    for sentence in sentences:
        int_sentence = []
        for character in sentence:
            int_sentence.append(vocab_int[character])
        int_sentences.append(int_sentence)
    return int_sentences

#Convert the sentences from integer to characters
def convert_integer_to_sentences(sentences,int_vocab):
    sentences_ch = []
    for row in sentences:
        sentence = []
        for index in row:
            sentence.append(int_vocab[index])
        sentences_ch.append(sentence)
    return sentences_ch

#Padding the data to have the same size
def padding_data(data, max_len,chars):
    padded_data = pad_sequences(data, padding='post', maxlen=max_len,value=len(chars))
    return padded_data

#Give noises in the data to create spelling mistakes
def noise_maker(sentence, threshold,vocab_int):
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass     
        i += 1
    return noisy_sentence

#Building X and Y to use for training
def building_x_and_y(data,treshhold,vocab_int):
    x=[]
    y=[]
    for sentence in data:
        x.append(noise_maker(sentence, treshhold,vocab_int))
        y.append(sentence)
    return x,y

#Build 3D sparse matrices to feed the LSTM neural network
def prepar_data_for_LSTM(x,nr_char):
    dim1=len(x)
    dim2=MAX_LEN_LINE
    dim3=nr_char+1
    x_lstm=np.zeros((dim1,dim2,dim3), dtype=np.bool)
    for idim1 in range(dim1-1):
        for idim2 in range(dim2):
            x_lstm[idim1,idim2,x[idim1,idim2]]=1
    return x_lstm

#Divide the dataset to training, validation, and test sets
def divide_data_to_train_valid_test(x):
    x_tr,x_test= train_test_split(x, test_size = 0.1, random_state = 2)
    x_tr,x_val= train_test_split(x_tr, test_size = 0.1, random_state = 2)
    return x_tr,x_val,x_test

#Compute the accuracy of the trained network using the test data
def validate_model_using_test(xtest,ytest,file_name):
    model = load_model('model_'+file_name+'.h5')
    test_loss, test_acc =model.evaluate(xtest, ytest)
    print('Accuracy: %f' % (test_acc*100))

#Generate LSTM model
def generate_model(output_len, chars,input_layers,hidden_size,dropout,initialization,output_layers):
    nr_features=len(chars)+1
    model = Sequential()
    for layer_number in range(input_layers):
        model.add(LSTM(hidden_size, input_shape=(None, nr_features)))
        model.add(Dropout(dropout))
    model.add(RepeatVector(output_len))
    for _ in range(output_layers):
        model.add(LSTM(hidden_size, return_sequences=True, kernel_initializer=initialization))
        model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(nr_features, kernel_initializer=initialization)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#train the model for all iteration steps
def iterate_training(model, X_train, y_train, X_val, y_val,size_batch,nr_epochs,nr_iterations,file_name):
    for iteration in range(1, nr_iterations):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        History=model.fit(X_train, y_train, batch_size=size_batch, epochs=nr_epochs)
        model.save('model_'+file_name+'.h5')
        model.save_weights('model_'+file_name+'_weights.h5')
        pd.DataFrame(History.history).to_csv("history.csv")

#Prepare the data and train the network        
def train_model(output_len, chars,input_layers,hidden_size,dropout,initialization,output_layers,size_batch,nr_epochs,nr_iterations,file_name):

    clean_data(DATA_FILENAME,CLEANDATA_FILENAME)
    data_cleaned=load_clean_data(CLEANDATA_FILENAME)
    proper_sentences=extract_proper_sentences(data_cleaned)
    vocab_to_int,int_to_vocab=build_dictionary_of_characters(VALID_CHARS)
    integer_sentences=convert_sentences_to_integer(proper_sentences,vocab_to_int)
    X,Y=building_x_and_y(integer_sentences,0.9,vocab_to_int)
    X_padded=padding_data(X, MAX_LEN_LINE,VALID_CHARS)
    Y_padded=padding_data(Y, MAX_LEN_LINE,VALID_CHARS)
    X_ready=prepar_data_for_LSTM(X_padded,len(VALID_CHARS))
    Y_ready=prepar_data_for_LSTM(Y_padded,len(VALID_CHARS))
    X_train,X_valid,X_test=divide_data_to_train_valid_test(X_ready)
    Y_train,Y_valid,Y_test=divide_data_to_train_valid_test(Y_ready)
    model=generate_model(output_len, chars,input_layers,hidden_size,dropout,initialization,output_layers)
    iterate_training(model, X_train, Y_train, X_valid, Y_valid,size_batch,nr_epochs,nr_iterations,file_name)
    validate_model_using_test(X_test,Y_test,file_name)

#Evaluate a tained network for a given data by the user    
def evaluate_trained_model(example_file_name,file_name,embedded=None):
    clean_data(example_file_name,'example_clean.txt')
    X_given=load_clean_data('example_clean.txt')
    vocab_to_int,int_to_vocab=build_dictionary_of_characters(VALID_CHARS)
    X_given=extract_proper_sentences(X_given)
    integer_sentences=convert_sentences_to_integer(X_given,vocab_to_int)
    X_padded=padding_data(integer_sentences, MAX_LEN_LINE,VALID_CHARS)

    if (embedded):
        X_ready=X_padded
    else:
        X_ready=prepar_data_for_LSTM(X_padded,len(VALID_CHARS))
    model = load_model('model_'+file_name+'.h5')
    Pred_sentences=[]
    for row in X_ready:
        if (embedded):
            row_new=row.reshape(1,len(row))
        else:
            row_new=row.reshape(1,len(row),len(row[0]))
        pred=model.predict_classes(row_new, verbose=0)
        pred=convert_integer_to_sentences(pred,int_to_vocab)
        pred=''.join(pred[0])
        pred = ''.join([char for char in pred if char in VALID_CHARS])
        Pred_sentences.append(pred)

    print("The results are:",Pred_sentences)