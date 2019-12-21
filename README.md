![](https://github.com/duncanmazza/ml_stock_prediction_api/workflows/Build%20Status/badge.svg)

Project for Machine Leraning: api for predicting stock prices

## Project Status: IN PROGRESS 

TODO: finish README

Install requirements:
```shell script
pip install -r requirements.txt
```

**Make sure that you are using Python 3.6 or later!** Our code is not backward compatable because we make use of type declarations in 3.6.

**TO BUILD THE DOCS LOCALLY go through the docs README [here](docsrc/README.md)**

The API is interactive, and runs off of a Bokeh server. To run it, open terminal and move to the root of the tool. Then, run the following line::

	bokeh serve --show .

This will open a server on http://localhost:5006/ml_stock_prediction_api. Open this link in your browser to access the API.


To run the LSTM code, run:
```shell script
python src/StockRNN.py
```

The code must be run from the root of the repository, or the PYTHONPATH enviornment variable should be set to the root of the repository. Additionally, to pull data from the Yahoo finance data api, you must have an internet connection for at least the first time you request a dataset (each query's returned data is cached in .cache/)

The main routine of `src/StockRNN` as it currently stands will train the model for a specified number of epochs and plot the results of the last forward pass on the test data. 
