"""
Tests the :py:mod:`src.get_data` module

@author: Duncan Mazza
"""

from src.get_data.pandas_stock_data import save_to_csv, numpy_array_of_company_daily_stock_close
import pytest
import os
from warnings import warn
from datetime import datetime
import numpy as np


class BColors:
    r"""
    Struct of ANSI escape sequences
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def test_save_to_csv_saves_file_no_csv_suffix(test_file_path="obscure_file_name", remove_test_file: bool = True):
    """
    Tests whether :py:func:`src.get_data.pandas_stock_data.save_to_csv` saves a file;

    :param test_file_path: ``file_path`` argument to :py:func:`src.get_data.pandas_stock_data.save_to_csv`
    :param remove_test_file (optional): set to True if the test file that is generated is desired to be removed
    :return:
    """
    save_to_csv(['APPL'], datetime(2010, 1, 1), datetime(2010, 1, 5), file_path=test_file_path)
    expected_output_file = os.path.join(os.getcwd(), test_file_path + ".csv")
    try:
        assert os.path.exists(expected_output_file)
    except AssertionError:
        warn(BColors.WARNING + "Failed to save file to expected output file - there be a file saved that was not "
                               "automatically removed by testing")
        raise AssertionError
    if remove_test_file:
        os.remove(expected_output_file)


def test_save_to_csv_saves_file_with_csv_suffix(test_file_path="obscure_file_name.csv", remove_test_file: bool = True):
    """
    Tests whether :py:func:`src.get_data.pandas_stock_data.save_to_csv` saves a file;

    :param test_file_path: ``file_path`` argument to :py:func:`src.get_data.pandas_stock_data.save_to_csv`
    :param remove_test_file (optional): set to True if the test file that is generated is desired to be removed
    :return:
    """
    save_to_csv(['APPL'], datetime(2010, 1, 1), datetime(2010, 1, 5), file_path=test_file_path)
    expected_output_file = os.path.join(os.getcwd(), test_file_path)
    try:
        assert os.path.exists(expected_output_file)
    except AssertionError:
        warn(BColors.WARNING + "Failed to save file to expected output file - there be a file saved that was not "
                               "automatically removed by testing")
        raise AssertionError
    if remove_test_file:
        os.remove(expected_output_file)


def test_numpy_array_of_company_daily_stock_close():
    stock_data = numpy_array_of_company_daily_stock_close('APPL', datetime(2017, 2, 9), datetime(2017, 2, 11))
    assert type(stock_data) == np.ndarray
    assert len(stock_data) == 2
    assert np.array_equiv(stock_data, np.array([132.42, 132.12]))

if __name__ == "__main__":
    pytest.main()
