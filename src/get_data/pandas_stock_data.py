"""
Utility functions for acquiring financial data

@author: Duncan Mazza
"""

import pandas_datareader.data as web
from pandas import DataFrame
from src.get_data.config import AV_API_KEY
from tests.BColors import BColors
import os
from datetime import datetime
import numpy as np
import requests


class MissingAPIKeyError(Exception):
    """
    TODO: documentation here
    """
    def __init__(self, api_key_name, message="You have left the default value in <root>/src/get_data/config.py for an "
                                             "API key unchanged; make sure that you fill this out so you can access "
                                             "your data."):
        Exception.__init__(self)
        print(message + " Missing API key: {}".format(api_key_name))


def return_stock_data(ticker: str, start_date: datetime, end_date: datetime, src: str = 'yahoo') -> DataFrame:
    r"""
    Returns a dataframe of stock data acquired using pandas_datareader.data. View more information
    `here <https://pandas-datareader.readthedocs.io/en/latest/remote_data.html>`__.


    :param ticker: ticker or list of tickers of companies to get stock data from
    :param start_date: datetime object of date for the start date of the stock data
    :param end_date: datetime object of the date for the end date of the stock data (note that the date range is
        inclusive with the lower bound start_date and exclusive with the upper bound end_date)
    :param src: string of data source; options include:

        From Alpha Vantage (note that there is a limit on API calls)
            * ``av-intraday`` - Intraday Time Series
            * ``av-daily`` - Daily Time Series
            * ``av-daily-adjusted`` - Daily Time Series (Adjusted)
            * ``obj:`av-weekly`` - Weekly Time Series
            * ``av-weekly-adjusted`` - Weekly Time Series (Adjusted)
            * ``av-monthly`` - Monthly Time Series
            * ``av-monthly-adjusted`` - Monthly Time Series (Adjusted)

            View more information here: `Alpha Vantage <https://www.alphavantage.co/documentation>`__

        From Yahoo:
            * ``yahoo``

    :return: pandas DataFrame of stock data
    """
    try:
        if src.__contains__("av-"):
            if AV_API_KEY == "YOUR_KEY_HERE":
                raise MissingAPIKeyError("AV_API_KEY")
            f = web.DataReader(ticker, src, start=start_date, end=end_date, api_key=AV_API_KEY)
        else:  # src == 'yahoo'
            f = web.get_data_yahoo(ticker, start_date, end_date)
    except KeyError:
        print(BColors.FAIL + "There was an error accessing data for the ticker {}".format(ticker) + BColors.WHITE)
        raise Exception
    except requests.exceptions.SSLError:
        print(BColors.FAIL + "A 'requests.exceptions.SSLError' was raised, which may be indicative of a lack of "
                             "internet connection; try again after verifying that you have a successful internet "
                             "connection." + BColors.WHITE)
        raise requests.exceptions.SSLError
    except requests.exceptions.ConnectionError:
        print(BColors.FAIL + "A 'requests.exceptions.ConnectionError' was raised, which may be indicative of a lack of "
                             "internet connection; try again after verifying that you have a successful internet "
                             "connection." + BColors.WHITE)
        raise requests.exceptions.ConnectionError
    return f


def save_to_csv(tickers: [str, ], start_date: datetime, end_date: datetime, src: str = 'yahoo',
                file_path: str = "stock_data.csv"):
    r"""
    # TODO: documentation here

    :param tickers: list of tickers of companies to get stock data from
    :param start_date: datetime object for the start date of the stock data
    :param end_date: datetime object for the end date of the stock data
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


def numpy_array_of_company_daily_stock_close_av(ticker: str, start_date: datetime, end_date: datetime) -> np.ndarray:
    """
    Returns daily stock of a company using the alpha-vantage api
    :param ticker:
    :param start_date:
    :param end_date:
    :return:
    """
    return np.array(return_stock_data(ticker, start_date, end_date, src='av_daily')['close'])

def numpy_array_of_company_daily_stock_close_yahoo(ticker: str, start_date: datetime, end_date: datetime) -> np.ndarray:
    """
    # TODO: documentation here
    :param ticker:
    :param start_date:
    :param end_date:
    :return:
    """
    return np.array(return_stock_data(ticker, start_date, end_date, src='yahoo')['Close'])

if __name__ == "__main__":
    stock_data = numpy_array_of_company_daily_stock_close_yahoo('IBM', datetime(2017, 2, 9), datetime(2017, 2, 11))
    print('here')