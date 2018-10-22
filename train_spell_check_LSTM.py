from Spell_check import *

if __name__ == '__main__':
    input_layers = 1
    output_layers = 1
    dropout = 0.2
    hidden_size = 200
    initialization = "he_normal"
    size_batch = 100
    nr_epochs = 10
    epochs = 10 
    nr_iterations=3
    train_model(MAX_LEN_LINE, VALID_CHARS,input_layers,hidden_size,dropout,initialization,output_layers,size_batch,nr_epochs,nr_iterations,'LSTM')