"""
Tests the :py:mod:`src.get_data` module

@author: Duncan Mazza
"""

from src.get_data import Company
from tests.BColors import BColors
import pytest
import os
from warnings import warn
from datetime import datetime
import numpy as np


@pytest.fixture()
def company():
    r"""
    TODO: documentation
    """
    return Company("IBM", start_date=datetime(2017, 2, 9), end_date=datetime(2017, 2, 13))


def test_save_to_csv_saves_file_no_csv_suffix(company: Company, test_file_path="obscure_file_name",
                                              remove_test_file: bool = True):
    r"""
    Tests whether :py:class:`src.get_data.pandas_stock_data.Company.save_to_csv` saves a file whose file name is not
    supplied with a ``.csv`` suffix

    :param company:
    :param test_file_path:
    :param remove_test_file:
    :return:
    """
    company.cache(test_file_path, company.data_frame)
    expected_output_file = os.path.join(os.getcwd(), test_file_path + ".csv")
    try:
        assert os.path.exists(expected_output_file)
    except AssertionError:
        warn(BColors.WARNING + "Failed to save file to expected output file - there be a file saved that was not "
                               "automatically removed by testing" + BColors.WHITE)
        raise AssertionError
    if remove_test_file:
        os.remove(expected_output_file)


def test_save_to_csv_saves_file_with_csv_suffix(company: Company, test_file_path: str = "obscure_file_name.csv",
                                                remove_test_file: bool = True):
    r"""
    Tests whether :py:class:`src.get_data.pandas_stock_data.Company.save_to_csv` saves a file whose file name is
    supplied with a ``.csv`` suffix
    :param company:
    :param test_file_path:
    :param remove_test_file:
    :return:
    """
    company.cache(test_file_path, company.data_frame)
    expected_output_file = os.path.join(os.getcwd(), test_file_path)
    try:
        assert os.path.exists(expected_output_file)
    except AssertionError:
        warn(BColors.WARNING + "Failed to save file to expected output file - there be a file saved that was not "
                               "automatically removed by testing" + BColors.WHITE)
        raise AssertionError
    if remove_test_file:
        os.remove(expected_output_file)


def test_return_numpy_array_of_company_daily_stock_close(company: Company):
    r"""
    TODO: documentation

    :param company:
    :return:
    """
    stock_data = company.return_numpy_array_of_company_daily_stock_close()
    assert type(stock_data) == np.ndarray
    assert len(stock_data) == 3
    compare_array = np.array([177.21000671, 178.67999268, 179.36000061])
    # typecast to 32bit integers to avoid discrepancy with comparing floating point numbers
    assert np.array_equal(stock_data.astype(np.int32), compare_array.astype(np.int32))


def return_numpy_array_of_company_daily_stock_percent_change(company: Company):
    r"""
    TODO: documentation

    :param company:
    :return:
    """
    stock_data = company.percent_change_of_time_series()
    assert type(stock_data) == np.ndarray
    assert len(stock_data) == 2
    compare_array = np.array([0.00829516, 0.00380573])
    # typecast to 32bit integers to avoid discrepancy with comparing floating point numbers
    assert np.array_equal(stock_data.astype(np.int32), compare_array.astype(np.int32))


if __name__ == "__main__":
    pytest.main()
