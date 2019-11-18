import pytest
from src.rnn.net_train import StockRNN
import numpy as np
import torch

@pytest.fixture()
def model_fixture():
    return StockRNN(1, 1, 1, 'IBM', auto_populate=False, src='yahoo')  # TODO: update these parameters

def test_populate_daily_stock_data(model_fixture: StockRNN):
    model_fixture.populate_daily_stock_data()
    time_delta = model_fixture.end_date - model_fixture.start_date
    assert len(model_fixture.daily_stock_data) % model_fixture.sequence_segment_length == 0

def test_populate_test_train(model_fixture: StockRNN):
    model_fixture.populate_daily_stock_data()
    model_fixture.populate_test_train(rand_seed=1)
    assert np.array_equal(model_fixture.test_sample_indices, np.array([5, 8, 9, 11, 12], dtype=np.int64))
    assert np.array_equal(model_fixture.train_sample_indices, np.array([14, 13, 17, 3, 21, 10, 18, 19, 4, 2, 20, 6, 7,
                                                                          22, 1, 16, 0, 15, 24, 23]))
    assert model_fixture.train_set.shape == torch.Size([20, 10])
    assert model_fixture.test_set.shape == torch.Size([5, 10])

if __name__ == "__main__":
    pytest.main()