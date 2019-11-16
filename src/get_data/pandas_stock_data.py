"""
Utility functions for acquiring financial data

@author: Duncan Mazza
"""

import pandas_datareader.data as web
from pandas import DataFrame
from src.get_data.config import AV_API_KEY
import os
from datetime import datetime


class MissingAPIKeyError(Exception):
    def __init__(self, api_key_name, message="You have left the default value in <root>/src/get_data/config.py for an "
                                             "API key unchanged; make sure that you fill this out so you can access "
                                             "your data."):
        Exception.__init__(self)
        print(message + " Missing API key: {}".format(api_key_name))


def return_stock_data(tickers: [str, ], start_date: datetime, end_date: datetime, src: str = 'av_daily'):
    r"""
    :param tickers: list of tickers of companies to get stock data from
    :param start_date: datetime object of date for the start date of the stock data
    :param end_date: datetime object of the date for the end date of the stock data
    :param src: string of data source; options include:

         * ``av-intraday`` - Intraday Time Series
         * ``av-daily`` - Daily Time Series
         * ``av-daily-adjusted`` - Daily Time Series (Adjusted)
         * ``av-weekly`` - Weekly Time Series
         * ``av-weekly-adjusted`` - Weekly Time Series (Adjusted)
         * ``av-monthly`` - Monthly Time Series
         * ``av-monthly-adjusted`` - Monthly Time Series (Adjusted)
         View more information here: `Alpha Vantage <https://www.alphavantage.co/documentation>`__
    :return: pandas dataframe of stock data
    """
    if AV_API_KEY == "YOUR_KEY_HERE":
        raise MissingAPIKeyError("AV_API_KEY")
    f: DataFrame
    f = web.DataReader("AAPL", "av-daily", start=datetime(2017, 2, 9), end=datetime(2017, 5, 24), api_key=AV_API_KEY)
    return f

def save_to_csv(tickers: [str, ], start_date: str, end_date: str, src: str = '',
                file_path: str = "stock_data.csv"):
    r"""
    Saves

    :param tickers: list of tickers of companies to get stock data from
    :param start_date: date string for the start date of the stock data (i.e. '2010-01-01')
    :param end_date: date string for the end date of the stock data (i.e. '2010-01-01')
    :param src: see :py:func:`return_stock_data` ``src`` argument description
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
