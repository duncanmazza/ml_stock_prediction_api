import pytest
from src.rnn.StockRNN import StockRNN
import numpy as np
import torch
from datetime import datetime


@pytest.fixture()
def _stock_rnn():
    r"""
    TODO: documentation
    """
    return StockRNN("IBM", 1, 1, start_date=datetime(2017, 1, 1), end_date=datetime(2018, 1, 1))


def test_populate_daily_stock_data(_stock_rnn: StockRNN):
    r"""
    TODO: documentation
    """
    assert len(_stock_rnn.daily_stock_data) % _stock_rnn.sequence_segment_length == 0


def test_populate_test_train_creates_correct_number_of_randomly_ordered_segments(_stock_rnn: StockRNN):
    r"""
    TODO: documentation
    """
    _stock_rnn.populate_daily_stock_data()
    _stock_rnn.populate_test_train(rand_seed=1)
    assert np.array_equal(_stock_rnn.test_sample_indices, np.array([5, 8, 9, 11, 12], dtype=np.int64))
    assert np.array_equal(_stock_rnn.train_sample_indices, np.array([14, 13, 17, 3, 21, 10, 18, 19, 4, 2, 20, 6, 7,
                                                                     22, 1, 16, 0, 15, 24, 23]))
    assert _stock_rnn.train_set.__len__() == 20
    assert _stock_rnn.test_set.__len__() == 5


# TODO: add more tests!


if __name__ == "__main__":
    pytest.main()
