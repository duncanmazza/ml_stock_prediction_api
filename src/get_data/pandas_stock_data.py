"""
Utility functions for acquiring financial data

@author: Duncan Mazza
"""

import pandas_datareader.data as web
from pandas import DataFrame
from tests.BColors import BColors
import os
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


class Company:
    def __init__(self, ticker: str, parent, populate_dataframe: bool = True):
        self.ticker = ticker
        self.parent = parent
        self.data_frame: DataFrame = DataFrame()
        self.initial_value = 0
        if populate_dataframe:
            self.populate_dataframe()

    def populate_dataframe(self):
        r"""
        TODO: Update documentation

        Returns a dataframe of stock data acquired using pandas_datareader.data. View more information
        `here <https://pandas-datareader.readthedocs.io/en/latest/remote_data.html>`__.

        :param populate_self_f:
        """
        try:
            data_frame = web.get_data_yahoo(self.parent.ticker, self.parent.start_date, self.parent.end_date)
        except KeyError:
            print(BColors.FAIL + "There was an error accessing data for the ticker {}".format(self.parent.ticker) + BColors.WHITE)
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
        self.data_frame = data_frame

    def save_to_csv(self, file_path: str = "stock_data.csv"):
        r"""
        TODO: documentation here

        :param tickers: list of tickers of companies to get stock data from
        :param start_date: datetime object for the start date of the stock data
        :param end_date: datetime object for the end date of the stock data
        :param src: see :py:func:`return_stock_data` ``src`` argument description
        :param file_path: string of the absolute path to save te file to (if it doesn't end with ``.csv``, then ``.csv``
         will be appended to the ``filepath`` string; if the provided path is not an absolute path, then it will be appended
         to ``os.getcwd()``.
        """
        if self.data_frame.empty:
            self.populate_dataframe()
        if not file_path.endswith(".csv"):
            file_path += ".csv"
        if not os.path.abspath(file_path):
            file_path = os.getcwd() + file_path
        self.data_frame.to_csv(file_path)

    def return_numpy_array_of_company_daily_stock_close(self) -> np.ndarray:
        r"""
        Returns daily stock of a company using the alpha-vantage api

        :param ticker:
        :param start_date:
        :param end_date:
        :return:
        """
        if self.data_frame.empty:
            self.populate_dataframe()
        return np.array(self.data_frame["Close"])

    def return_numpy_array_of_company_daily_stock_percent_change(self) -> np.ndarray:
        """
        TODO: documentation here

        :param src:
        :param ticker:
        :param start_date:
        :param end_date:
        :return:
        """
        daily_stock_data = self.return_numpy_array_of_company_daily_stock_close()
        start_array: np.ndarray = daily_stock_data[:-1]
        end_array: np.ndarray = daily_stock_data[1:]
        return (end_array - start_array) / start_array

# if __name__ == "__main__":
    # stock_data = numpy_array_of_company_daily_stock_close_yahoo('IBM', datetime(2017, 2, 9), datetime(2017, 2, 11))