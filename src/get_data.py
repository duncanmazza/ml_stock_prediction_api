"""
Utility functions for acquiring financial data

@author: Duncan Mazza
"""

import pandas_datareader.data as web
from pandas import DataFrame, read_csv, errors
import os
import numpy as np
import requests
from datetime import datetime

ZERO_TIME = " 00:00:00"


class Company:
    def __init__(self, ticker: str, start_date: datetime, end_date: datetime, call_populate_dataframe: bool = True,
                 cache_bool: bool = True,):
        """
        TODO: documentation here

        :param ticker:
        :param start_date:
        :param end_date:
        :param call_populate_dataframe:
        :param cache_bool:
        """
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.data_frame: DataFrame = DataFrame()
        self.cache_bool = cache_bool
        self.end_date_changed = False
        self.start_date_changed = False

        if call_populate_dataframe:
            self.populate_dataframe()

    def populate_dataframe(self):
        r"""
        Populates :attr:`data_frame` with stock data acquired using pandas_datareader.data. View more information
        `here <https://pandas-datareader.readthedocs.io/en/latest/remote_data.html>`__. Modifies :attr:`start_date`,
        :attr:`start_date_changed`, :attr:`end_date`, and :attr:`end_date_changed` if :attr:`start_date` and/or
        :attr:`end_date` are different than the actual start and end dates in :attr:`data_frame` such that
        :attr:`start_date` and :attr:`end_date` equal the actual start and end dates in :attr:`data_frame` (and
        :attr:`start_date_changed` and :attr:`end_date_changed` reflect whether :attr:`start_date` and :attr:`end_date`
        were changed respectively).
        """
        if self.ticker == "dummy":
            self.data_frame = self.return_dummy_data()
        else:
            self.data_frame = self.return_data()
        data_frame_start_date = self.data_frame["Date"][0]
        data_frame_end_date = self.data_frame["Date"][self.data_frame.last_valid_index()]
        if data_frame_start_date != self.start_date:
            self.start_date = data_frame_start_date
            self.start_date_changed = True
        if data_frame_end_date != self.end_date:
            self.end_date = data_frame_end_date
            self.end_date_changed = True

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
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        if ticker is None:
            ticker = self.ticker

        start_date_str = start_date.__str__().strip(ZERO_TIME)
        end_date_str = end_date.__str__().strip(ZERO_TIME)
        rel_file_path = os.path.join(".cache", "&".join([ticker,
                                                         start_date_str,
                                                         end_date_str])) + ".csv"
        if os.path.exists(os.path.join(os.getcwd(), rel_file_path)):
            try:
                data_frame = read_csv(os.path.join(os.getcwd(), rel_file_path))
                print(
                    " > Loaded data requested for {} from {} to {} from '.cache/' folder".format(ticker, start_date_str,
                                                                                         end_date_str))
                return data_frame
            except errors.ParserError:
                print("Could not load data for {} from {} to {} from .cache/ folder (although the path exists"
                      .format(ticker, start_date, end_date))
                pass

        try:
            data_frame = web.get_data_yahoo(ticker, start_date, end_date)
            print(" > Loaded data requested for {} from {} to {} from internet".format(ticker, start_date_str,
                                                                                       end_date_str))
        except requests.exceptions.SSLError:
            print("ERROR: A 'requests.exceptions.SSLError' was raised, which may be indicative of a lack of "
                  "internet connection; try again after verifying that you have a successful internet "
                  "connection.")
            raise requests.exceptions.SSLError
        except requests.exceptions.ConnectionError:
            print("ERROR: A 'requests.exceptions.ConnectionError' was raised, which may be indicative of a "
                  "lack of internet connection; try again after verifying that you have a successful "
                  "internet connection.")
            raise requests.exceptions.ConnectionError

        if self.cache_bool:
            self.cache(rel_file_path, data_frame)

        # loading the dataframe from the internet as opposed to from the csv cache results in a different handling of
        # the timestamp index, where the timestamp index is converted to a "Date" column when cached. Consequently,
        # a "Date" column needs to be inserted
        data_frame.insert(0, "Date", data_frame.index)
        data_frame.index = np.arange(0, len(data_frame), 1)

        return data_frame

    def return_dummy_data(self):
        """
        Creates linear stock data as dummy data for testing a model

        :return: numpy array of dummy data
        """
        return DataFrame({"Date" : [i for i in range(200)], "Close" : [i for i in range(200)]})

    def revise_start_date(self, new_start_date: datetime):
        """
        Modifies :attr:`data_frame` such that the starting date of the data is equal to ``new_start_date`` (all prior
        data is deleted).

        :param new_start_date: a datetime object of the new start date for :attr:`data_frame` (where ``new_start_date``
         exists and is unique in ``self.data_frame["Date"]``
        """
        loc = self.data_frame["Date"][self.data_frame["Date"] == new_start_date].index[0]
        self.data_frame = self.data_frame[loc:]
        self.start_date = new_start_date

    def revise_end_date(self, new_end_date: datetime):
        """
        Modifies :attr:`data_frame` such that the last date of the data is equal to ``new_end_date`` (all following
        data is deleted).

        :param new_end_date: a datetime object of the new end date for :attr:`data_frame` (where ``new_end_date``
         exists and is unique in ``self.data_frame["Date"]``
        """
        loc = self.data_frame["Date"][self.data_frame["Date"] == new_end_date].index[0]
        self.data_frame = self.data_frame[:loc + 1]  # add 1 so that the last date is included
        self.data_frame: DataFrame
        self.end_date = new_end_date

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

    def return_numpy_array_of_company_daily_stock_percent_change(self, rolling_avg_length: int = 4) -> np.ndarray:
        """
        Converts the numpy array of the closing stock data (acquired by calling
        :method:`return_numpy_array_of_company_daily_stock_close`) into an array of day-over-day percent change.
        Adds a value of 0 at the beginning of the array to maintain sequence length.

        :param apply_rolling_avg: if nonzero, applies a rolling average filter to the percent change data (see
        :method:`moving_average` for details on how the rolling average is calculated with padding)
        :return: numpy array of length 1 less than the array generated by
        :method:`return_numpy_array_of_company_daily_stock_close`
        """
        daily_stock_data = self.return_numpy_array_of_company_daily_stock_close()
        start_array: np.ndarray = daily_stock_data[:-1]
        end_array: np.ndarray = daily_stock_data[1:]
        percent_change = (end_array - start_array) / start_array
        # if rolling_avg_length == 0:
        return np.concatenate((np.array([0]), percent_change), axis=0)
        # else:
        #     return self.moving_average(percent_change, n=rolling_avg_length)

    def get_date_at_index(self, i):
        """
        Returns the datetime object at index

        :param i: index to return the date of
        """
        return self.data_frame["Date"].iloc[[i]].values[0]

    @staticmethod
    def moving_average(a, n, padding : bool =True):
        r"""
        Calculates the moving average of a one-dimensional numpy array ``a``, capable of utilizing a padding of
        length ``n - 1`` at the beginning of the pre-filtered data so that the length is not truncated; the padding is
        populated with the same value as the first value of the % change sequence.
        :param a: one-dimensional numpy array
        :param n: length of rolling average filter
        :param padding: boolean for whether to utilize padding
        :return: filtered array
        """
        if padding:
            a = np.concatenate((np.ones((n - 1,)) * a[0], a), axis=0)
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def reconstruct_stock_from_percent_change(self, percent_change_vec: np.ndarray, initial_condition_index: int):
        """
        Reconstruct the stock prices from percent change

        :param percent_change_vec: vector of percent changes
        :param initial_condition_index: index of initial condition for the % change
        """
        len_percent_change_vec = len(percent_change_vec)
        stock_price = np.zeros((len_percent_change_vec + 1))
        stock_price[0] = self.data_frame["Close"].iloc[[initial_condition_index]]
        for i in range(1, len_percent_change_vec + 1):
            stock_price[i] = stock_price[i-1] * (1 + percent_change_vec[i-1])
        return stock_price
