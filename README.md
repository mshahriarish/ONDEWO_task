# The objective:
To perform a simple spell check using a deep neural network.
## Methods
I used two different methods to accomplish the spell-check.

1. As the first Neural network, I used an LSTM neural network to performs the spell check.


2. As the second example, I introduced a pre-trained character based embedding layer for the input.

## What I Learned:

This was my very first time to work in NLP. I found the topic fascinating and challenging. The field has great potential. However, it needs extensive work to achieve its high potential. I learned how to convert text to the numbers and later to the proper format to feed our neural network. I briefly became familiar with the problems that NLP experts face on a daily basis. I gathered some information regarding the architecture and type of networks and hidden layers that we need to use in this field. I found the experience satisfying and challenging. I would be happy to work in this field. 

## To Do When I Have Time:

1. Train the neural network using a GPU.

2. Train the networks using different configurations to find the best configuration possible.

3. Work on the architecture of the network to achieve the best one.

4. Perform a more in-depth cleaning to consider more special characters.

5. Implement a combined (word-based, character-based) network to achive better results.

6. Adapt my network for German and Finish.

## Codes Information

The presented codes have two modules containing all the necesarry functions:

1. Spell_check

2. Spell_check_embedding

Moreover, there are four mains as follows:

1. train_spell_check_LSTM: Trains the network using an LSTM network and a text dataset name "news.txt"

2. train_spell_check_embedded: Trains a network which contains an embedding layer using text dataset name "news.txt"

3. evaluate_spell_check_LSTM: Performs spell correction on a given text named "example.txt" using LSTM network

4. evaluate_spell_check_embedded: Performs spell correction on a given text named "example.txt" using the embedding network
