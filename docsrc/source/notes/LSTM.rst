======================
Understanding the LSTM
======================
The first of our machine learning models is a long-term short-term memory (LSTM) network, which is a specific type of recurrent neural networks. Networks classified as recurrent neural networks allow information to persist between passes of data through the network; this is similar to how humans use memory to aid their immediate understanding of the world. As the name suggests, LSTMs have the capability to encode information about previous states - both recent and not - to inform their output (shown in diagram below). LSTMs are designed for sequential data, making them a logical choice for our application.
.. image:: lstm.png

LSTMs can process sequences of arbitrary length with any pre-defined number of features, and output a sequence with a different and arbitrary number of features. To leverage these attributes, we pass in not only sequences containing the stock of the company we wish to predict, but also the stock prices of other companies. While stock prices are inherently unpredictable, we built our model on the notion that there there may exist learn-able relationships between companies' stocks.

As a side note, evaluation mode prevents the dropout from occurring; a dropout probability of 0.3 was used.
