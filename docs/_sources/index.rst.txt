.. Documentation master file, created by
   sphinx-quickstart on Wed Nov 13 17:24:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Stock Prediction API documentation
=================================================

Github: https://github.com/duncanmazza/ml_stock_prediction_api/tree/master

Introduction
============

The goal of our project is to create a stock market modeling API that would utilize multiple different types of machine learning models to learn patterns in stock prices and make predictions about future stock prices. With the goal of an API in mind, we have integrated automatic documentation building and our code base consists primarily of object-oriented code. To that end, we integrated Yahoo's API a class that serves as a common ground for the disparate models to acquire data.

Information on each part of the machine learning model as well as examples of the results are found in the notes below.

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Notes

   notes/LSTM.rst
   notes/GPM.rst
   notes/combined.rst
   notes/usage.rst
   notes/examples.rst

All of our module's code is documented and can be found here:

.. toctree::
   :maxdepth: 2
   :caption: Code Documentation:

   _build/modules/


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
