from Spell_check import *
from keras.layers.embeddings import Embedding

embeddings_path = "glove.840B.300d-char.txt"
embedding_dim = 300
#Load a pre-trained chracter-based weights which was extracted from Glove word embedding
def build_trained_embedded_matrix(char_len,dic_vocab_to_int):
    embedding_vectors = {}
    with open(embeddings_path, 'r') as f:
        for line in f:
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            char = line_split[0]
            embedding_vectors[char] = vec

    embedding_matrix = np.zeros((char_len, 300))
    for char, i in dic_vocab_to_int.items():
        embedding_vector = embedding_vectors.get(char)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
#Generate the model
def generate_model_embedding(output_len, chars,input_layers,hidden_size,dropout,output_layers,embedding_mat):
    nr_features=len(chars)+1
    model = Sequential()
    model.add(Embedding(nr_features, embedding_dim, input_length=output_len,weights=[embedding_mat]))
    for i in range(input_layers):
        model.add(LSTM(hidden_size, activation='sigmoid'))
        model.add(Dropout(dropout))
        model.add(RepeatVector(output_len))
    
    for i in range(output_layers):
        model.add(LSTM(hidden_size,activation='relu'))
        model.add(Dropout(dropout))
        model.add(RepeatVector(output_len))
    model.add(TimeDistributed(Dense(nr_features)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#Train the model
def train_model_embedded(output_len, chars,input_layers,hidden_size,dropout,output_layers,size_batch,nr_epochs,nr_iterations,file_name):
    clean_data(DATA_FILENAME,CLEANDATA_FILENAME)
    data_cleaned=load_clean_data(CLEANDATA_FILENAME)
    vocab_to_int,int_to_vocab=build_dictionary_of_characters(VALID_CHARS)
    proper_sentences=extract_proper_sentences(data_cleaned)
    integer_sentences=convert_sentences_to_integer(proper_sentences,vocab_to_int)
    X,Y=building_x_and_y(integer_sentences,0.9,vocab_to_int)
    X_padded=padding_data(X, MAX_LEN_LINE,VALID_CHARS)
    Y_padded=padding_data(Y, MAX_LEN_LINE,VALID_CHARS)
    Y_ready=prepar_data_for_LSTM(Y_padded,len(VALID_CHARS))
    X_train,X_valid,X_test=divide_data_to_train_valid_test(X_padded)
    Y_train,Y_valid,Y_test=divide_data_to_train_valid_test(Y_ready)
    Embedding_matrix=build_trained_embedded_matrix(len(VALID_CHARS)+1,vocab_to_int)
    model=generate_model_embedding(MAX_LEN_LINE,VALID_CHARS,input_layers,hidden_size,dropout,output_layers,Embedding_matrix)
    iterate_training(model, X_train, Y_train, X_valid, Y_valid,size_batch,nr_epochs,nr_iterations,file_name)
    validate_model_using_test(X_test,Y_test,file_name)