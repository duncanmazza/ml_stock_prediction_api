"""
Utility functions for acquiring financial data

@author: Duncan Mazza
"""

import pandas_datareader.data as web
from pandas import DataFrame, read_csv, errors
from tests.BColors import BColors
import os
import numpy as np
import requests
from datetime import datetime


class Company:
    def __init__(self, ticker: str, parent, populate_dataframe: bool = True, cache_bool: bool = True):
        self.ticker = ticker
        self.parent = parent
        self.data_frame: DataFrame = DataFrame()
        self.initial_value = 0
        self.cache_bool = cache_bool
        if populate_dataframe:
            self.populate_dataframe()

    def return_data(self, ticker: str = None, start_date: datetime = None, end_date: datetime = None) -> DataFrame:
        """
        Returns the DataFrame containing the financial data for the prescribed company. This function will pull the
        data from the Yahoo API built into :ref:`pandas_datareader` if it has not been cached and will then cache the
        data, or it will read the data from the cached ``csv`` file. The cached files are named with the ticker, start
        date, and end dates that specify the API query, and exist in the ``.cache/`` folder located under the current
        working directory.

        :param ticker: ticker string for the company whose data will be retrieved
        :param start_date: start date for the data record
        :param end_date: end date for the data record
        :return: DataFrame of financial data
        """
        if start_date is None:
            start_date = self.parent.start_date
        if end_date is None:
            end_date = self.parent.end_date
        if ticker is None:
            ticker = self.parent.ticker

        rel_file_path = os.path.join(".cache", "&".join([ticker,
                                                         start_date.__str__().strip(" 00:00:00"),
                                                         end_date.__str__().strip(" 00:00:00")])) + ".csv"
        if os.path.exists(os.path.join(os.getcwd(), rel_file_path)):
            try:
                data_frame = read_csv(os.path.join(os.getcwd(), rel_file_path))
                print("Loaded data for {} from {} to {} from .cache/ folder".format(ticker, start_date, end_date))
                return data_frame
            except errors.ParserError:
                print("Could not load data for {} from {} to {} from .cache/ folder (although the path exists"
                      .format(ticker, start_date, end_date))
                pass

        try:
            data_frame = web.get_data_yahoo(ticker, start_date, end_date)
            print("Loaded data for {} from {} to {} from internet".format(ticker, start_date, end_date))
        except KeyError:
            print(BColors.FAIL + "There was an error accessing data for the ticker {}".format(self.parent.ticker) +
                  BColors.WHITE)
            raise KeyError
        except requests.exceptions.SSLError:
            print(BColors.FAIL + "A 'requests.exceptions.SSLError' was raised, which may be indicative of a lack of "
                                 "internet connection; try again after verifying that you have a successful internet "
                                 "connection." + BColors.WHITE)
            raise requests.exceptions.SSLError
        except requests.exceptions.ConnectionError:
            print(BColors.FAIL + "A 'requests.exceptions.ConnectionError' was raised, which may be indicative of a "
                                 "lack of internet connection; try again after verifying that you have a successful "
                                 "internet connection." + BColors.WHITE)
            raise requests.exceptions.ConnectionError

        if self.cache_bool:
            self.cache(rel_file_path, data_frame)

        return data_frame

    def populate_dataframe(self):
        r"""
        Populates :attr:`data_frame` with stock data acquired using pandas_datareader.data. View more information
        `here <https://pandas-datareader.readthedocs.io/en/latest/remote_data.html>`__.
        """
        self.data_frame = self.return_data()

    def cache(self, file_path: str, data_frame: DataFrame = None):
        """
        Saves a DataFrame as a ``.csv`` to a path relative to the current working directory.

        :param file_path: path to save the :ref:`DataFrame` to; if not an absolute path, then it is used as a path
        relative to the current working directory.
        :param data_frame: DataFrame to save (if not specified, will use :attr:`data_frame` (attribute)
        """
        if not file_path.endswith(".csv"):
            file_path += ".csv"
        if not os.path.abspath(file_path):
            file_path = os.getcwd() + file_path
        if not os.path.isdir(os.path.join(os.getcwd(), ".cache")):
            os.mkdir(os.path.join(os.getcwd(), ".cache"))
        if data_frame is None:
            data_frame = self.data_frame
        data_frame.to_csv(file_path)

    def return_numpy_array_of_company_daily_stock_close(self) -> np.ndarray:
        """
        Returns a numpy array of the "Close" column of :attr:`data_frame`.

        :return: numpy array of closing stock prices indexed by day
        """
        if self.data_frame.empty:
            self.populate_dataframe()
        return np.array(self.data_frame["Close"])

    def return_numpy_array_of_company_daily_stock_percent_change(self) -> np.ndarray:
        """
        Converts the numpy array of the closing stock data (acquired by calling
        :method:`return_numpy_array_of_company_daily_stock_close`) into an array of day-over-day percent change.

        :return: numpy array of length 1 less than the array generated by
        :method:`return_numpy_array_of_company_daily_stock_close`
        """
        daily_stock_data = self.return_numpy_array_of_company_daily_stock_close()
        start_array: np.ndarray = daily_stock_data[:-1]
        end_array: np.ndarray = daily_stock_data[1:]
        return (end_array - start_array) / start_array

# if __name__ == "__main__":
# stock_data = numpy_array_of_company_daily_stock_close_yahoo('IBM', datetime(2017, 2, 9), datetime(2017, 2, 11))
