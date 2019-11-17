"""
Tests the :py:mod:`src.get_data` module

@author: Duncan Mazza
"""

from src.get_data.pandas_stock_data import save_to_csv, numpy_array_of_company_daily_stock_close_av, \
    numpy_array_of_company_daily_stock_close_yahoo
from tests.BColors import BColors
import pytest
import os
from warnings import warn
from datetime import datetime
import numpy as np


def test_save_to_csv_saves_file_no_csv_suffix(test_file_path="obscure_file_name", remove_test_file: bool = True):
    """
    Tests whether :py:func:`src.get_data.pandas_stock_data.save_to_csv` saves a file;

    :param remove_test_file:
    :param test_file_path: ``file_path`` argument to :py:func:`src.get_data.pandas_stock_data.save_to_csv`
    :param remove_test_file (optional): set to True if the test file that is generated is desired to be removed
    """
    save_to_csv(['IBM'], datetime(2010, 1, 1), datetime(2010, 1, 5), file_path=test_file_path)
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
    """
    save_to_csv(['IBM'], datetime(2010, 1, 1), datetime(2010, 1, 5), file_path=test_file_path)
    expected_output_file = os.path.join(os.getcwd(), test_file_path)
    try:
        assert os.path.exists(expected_output_file)
    except AssertionError:
        warn(BColors.WARNING + "Failed to save file to expected output file - there be a file saved that was not "
                               "automatically removed by testing")
        raise AssertionError
    if remove_test_file:
        os.remove(expected_output_file)


# Uncomment following test when you want to test the Alpha-Vantage api (requires an API key; this test should only be
# run locally and not for CI).
# def test_numpy_array_of_company_daily_stock_close_av():
#     stock_data = numpy_array_of_company_daily_stock_close_av('APPL', datetime(2017, 2, 9), datetime(2017, 2, 11))
#     assert type(stock_data) == np.ndarray
#     assert len(stock_data) == 2
#     assert np.array_equiv(stock_data, np.array([132.42, 132.12]))


def test_numpy_array_of_company_daily_stock_close_yahoo():
    stock_data = numpy_array_of_company_daily_stock_close_yahoo('IBM', datetime(2017, 2, 9), datetime(2017, 2, 11))
    assert type(stock_data) == np.ndarray
    assert len(stock_data) == 2
    print('##########Stock data:', stock_data)
    compare_array = np.array([177.21000671, 178.67999268])
    # typcast to 32bit integers to avoid discrepancy with comparing floating point numbers
    assert np.array_equal(stock_data.astype(np.int32), compare_array.astype(np.int32))


if __name__ == "__main__":
    pytest.main()
