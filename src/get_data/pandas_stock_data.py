"""
Utility functions for acquiring financial data

@author: Duncan Mazza
"""

from pandas_datareader import data
from src.get_data.config import AV_API_KEY
import os


def return_stock_data(tickers: [str,], start_date: str, end_date: str, src: str ='yahoo'):
    r"""
    :param tickers: list of tickers of companies to get stock data from
    :param start_date: date string for the start date of the stock data (i.e. '2010-01-01')
    :param end_date: date string for the end date of the stock data (i.e. '2010-01-01')
    :param src: string of api to use (default: yahoo)
    :return: pandas dataframe of stock data
    """
    return data.DataReader(tickers, src, start_date, end_date)


def save_to_csv(tickers: [str,], start_date: str, end_date: str, src: str ='yahoo',
                file_path: str = "stock_data.csv"):
    r"""
    Saves

    :param tickers: list of tickers of companies to get stock data from
    :param start_date: date string for the start date of the stock data (i.e. '2010-01-01')
    :param end_date: date string for the end date of the stock data (i.e. '2010-01-01')
    :param src: string of api to use (default: yahoo)
    :param file_path: string of the absolute path to save te file to (if it doesn't end with ``.csv``, then ``.csv``
     will be appended to the ``filepath`` string; if the provided path is not an absolute path, then it will be appended
     to ``os.getcwd()``.
    """
    stock_data = return_stock_data(tickers, start_date, end_date, src)
    if not file_path.endswith(".csv"):
        file_path += ".csv"
    if not os.path.abspath(file_path):
        file_path = os.getcwd() + file_path
    stock_data.to_csv(file_path)

if __name__ == "__main__":
    stock_data = return_stock_data(['APPL'], '2010-01-01', '2010-01-05')
    print('here')