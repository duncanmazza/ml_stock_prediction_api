from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd


def return_stock_data(tickers: [str,], start_date: str, end_date: str, src: str ='yahoo'):
    r"""
    :param tickers: list of tickers of companies to get stock data from
    :param start_date: date string for the start date of the stock data (i.e. '2010-01-01')
    :param end_date: date string for the end date of the stock data (i.e. '2010-01-01')
    :param src: string of api to use (default: yahoo)
    :return: pandas dataframe of stock data
    """
    return data.DataReader(tickers, src, start_date, end_date)


def save_to_csv(tickers: [str,], start_date: str, end_date: str, src: str ='yahoo'):
    r"""
    Saves
    :param tickers: list of tickers of companies to get stock data from
    :param start_date: date string for the start date of the stock data (i.e. '2010-01-01')
    :param end_date: date string for the end date of the stock data (i.e. '2010-01-01')
    :param src: string of api to use (default: yahoo)
    :return: void
    """
    data = return_stock_data(tickers, start_date, end_date, src)
    # TODO save dataframe
