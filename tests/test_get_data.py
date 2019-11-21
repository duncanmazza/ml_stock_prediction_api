"""
Tests the :py:mod:`src.get_data` module

@author: Duncan Mazza
"""

from src.rnn.StockRNN import StockRNN
from tests.BColors import BColors
import pytest
import os
from warnings import warn
from datetime import datetime
import numpy as np


@pytest.fixture()
def _stock_rnn():
    r"""
    TODO: documentation
    """
    return StockRNN("IBM", 1, 1, start_date=datetime(2017, 2, 9), end_date=datetime(2017, 2, 12),
                    sequence_segment_length=1)


def test_save_to_csv_saves_file_no_csv_suffix(_stock_rnn: StockRNN, test_file_path="obscure_file_name",
                                              remove_test_file: bool = True):
    r"""
    Tests whether :py:class:`src.get_data.pandas_stock_data.Company.save_to_csv` saves a file whose file name is not
    supplied with a ``.csv`` suffix

    :param _stock_rnn:
    :param test_file_path:
    :param remove_test_file:
    :return:
    """
    _stock_rnn.coi.save_to_csv(file_path=test_file_path)
    expected_output_file = os.path.join(os.getcwd(), test_file_path + ".csv")
    try:
        assert os.path.exists(expected_output_file)
    except AssertionError:
        warn(BColors.WARNING + "Failed to save file to expected output file - there be a file saved that was not "
                               "automatically removed by testing" + BColors.WHITE)
        raise AssertionError
    if remove_test_file:
        os.remove(expected_output_file)


def test_save_to_csv_saves_file_with_csv_suffix(_stock_rnn: StockRNN, test_file_path: str = "obscure_file_name.csv",
                                                remove_test_file: bool = True):
    r"""
    Tests whether :py:class:`src.get_data.pandas_stock_data.Company.save_to_csv` saves a file whose file name is
    supplied with a ``.csv`` suffix
    :param _stock_rnn:
    :param test_file_path:
    :param remove_test_file:
    :return:
    """
    _stock_rnn.coi.save_to_csv(file_path=test_file_path)
    expected_output_file = os.path.join(os.getcwd(), test_file_path)
    try:
        assert os.path.exists(expected_output_file)
    except AssertionError:
        warn(BColors.WARNING + "Failed to save file to expected output file - there be a file saved that was not "
                               "automatically removed by testing" + BColors.WHITE)
        raise AssertionError
    if remove_test_file:
        os.remove(expected_output_file)


def test_return_numpy_array_of_company_daily_stock_close(_stock_rnn: StockRNN):
    r"""
    TODO: documentation

    :param _stock_rnn:
    :return:
    """
    stock_data = _stock_rnn.coi.return_numpy_array_of_company_daily_stock_close()
    assert type(stock_data) == np.ndarray
    assert len(stock_data) == 3
    compare_array = np.array([177.21000671, 178.67999268, 179.36000061])
    # typecast to 32bit integers to avoid discrepancy with comparing floating point numbers
    assert np.array_equal(stock_data.astype(np.int32), compare_array.astype(np.int32))


def return_numpy_array_of_company_daily_stock_percent_change(_stock_rnn: StockRNN):
    r"""
    TODO: documentation

    :param _stock_rnn:
    :return:
    """
    stock_data = _stock_rnn.coi.return_numpy_array_of_company_daily_stock_percent_change()
    assert type(stock_data) == np.ndarray
    assert len(stock_data) == 2
    compare_array = np.array([0.00829516, 0.00380573])
    # typecast to 32bit integers to avoid discrepancy with comparing floating point numbers
    assert np.array_equal(stock_data.astype(np.int32), compare_array.astype(np.int32))


if __name__ == "__main__":
    pytest.main()
