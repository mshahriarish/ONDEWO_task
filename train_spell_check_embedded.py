from Spell_check import *
from Spell_check_embedding import *

if __name__ == '__main__':
    input_layers = 2
    output_layers = 3
    dropout = 0.2
    hidden_size = 200
    size_batch = 100
    nr_epochs = 10
    epochs = 10 
    nr_iterations=3
    train_model_embedded(MAX_LEN_LINE, VALID_CHARS,input_layers,hidden_size,dropout,output_layers,size_batch,nr_epochs,nr_iterations,'embedding')