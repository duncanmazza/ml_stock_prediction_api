"""
Code to train the RNN

@author: Duncan Mazza
"""

from torch import Tensor
from tests.BColors import BColors
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from datetime import datetime
from src.get_data import Company
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time

DEVICE = "cuda"  # selects the gpu to be used
TO_GPU_FAIL_MSG = BColors.FAIL + "Unable to successfully run model.to('{}'). If running in Collaboratory, make sure " \
                                 "that you have enabled the GPU your settings".format(DEVICE) + BColors.WHITE


class StockRNN(nn.Module):
    r"""
    Class for handling all RNN operations.
    """
    train_set: TensorDataset
    test_set: TensorDataset
    train_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, ticker: str, lstm_hidden_size: int, lstm_num_layers: int, to_compare: [str, ] = None,
                 start_date: datetime = datetime(2017, 1, 1), end_date: datetime = datetime(2018, 1, 1),
                 sequence_segment_length: int = 10, drop_prob: float = 0.5, device: str = DEVICE,
                 auto_populate: bool = True, train_data_prop: float = 0.8, lr: float = 1e-4,
                 train_batch_size: int = 2, test_batch_size: int = 2, num_workers: int = 2, label_length: int = 5):
        r"""
        TODO: documentation here

        :param lstm_hidden_size:
        :param lstm_num_layers:
        :param ticker:
        :param to_compare:
        :param start_date:
        :param end_date:
        :param sequence_segment_length:
        :param drop_prob:
        :param device:
        :param auto_populate:
        :param train_data_prop:
        :param lr:
        :param train_batch_size:
        :param test_batch_size:
        :param num_workers:
        :param label_length:
        """
        super(StockRNN, self).__init__()

        # variable indicating success of calling self.to(DEVICE), where 0 indicates that it hasn't been tried yet, -1 
        # indicates that it failed, and 1 indicates that it was successful  
        self.__togpu_works__ = 0

        # __init__ params
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.drop_prob = drop_prob
        self.device = device
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_segment_length = sequence_segment_length
        self.auto_populate = auto_populate
        self.train_data_prop = train_data_prop
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        if label_length >= self.sequence_segment_length:
            print("Label length was specified to be {}, but cannot be >= self.sequence_segment_length; setting "
                  "self.label_length to self.sequence_segment_length - 1.")
            self.label_length = self.sequence_segment_length - 1
        else:
            self.label_length = label_length
        self.to_compare = []

        # company of interest
        self.coi = Company(self.ticker, self)

        if to_compare is not None:
            for company in to_compare:
                self.to_compare.append(Company(company, self))

        # initialize objects used during forward pass
        self.lstm = nn.LSTM(1, self.lstm_hidden_size, self.lstm_num_layers, dropout=self.drop_prob)
        # self.dropout = nn.Dropout(drop_prob)
        self.fc_after_lstm = nn.Linear(self.lstm_hidden_size, 1)

        # initialize attributes with placeholder arrays
        self.daily_stock_data = np.array(0)
        self.train_sample_indices = np.array(0)
        self.test_sample_indices = np.array(0)
        self.train_loader_len = 0
        self.test_loader_len = 0

        # initialize optimizer and loss
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.auto_populate:
            self.populate_daily_stock_data()
            self.populate_test_train()
            self.populate_loaders()

    def __togpu__(self, successful):
        r"""
        Sets the value of :attr:`__togpu_works__`, which is used in such a way that expensive error catching isn't run
        every epoch of training.

        :param successful: boolean for whether ``.to(gpu)`` was called successfully
        """
        if successful:
            self.__togpu_works__ = 1
        else:
            self.__togpu_works__ = -1

    def peek_dataset(self, figsize: (int, int) = (10, 5)):
        r"""
        Creates a simple line plot of the stock data

        :param figsize: tuple of integers for :class:`plt.subplots` ``figsize`` argument
        """
        _, axs = plt.subplots(1, 1, figsize=figsize)
        axs: Axes
        axs.plot(self.daily_stock_data)
        axs.set_title("Ticker {} from {} to {}".format(self.coi.ticker, self.start_date, self.end_date))
        axs.set_xlabel("Time")
        axs.set_ylabel("Price (USD)")
        plt.show()

    def populate_daily_stock_data(self, truncate: bool = True):
        r"""
        Populates ``self.daily_stock_data`` using relevant class attributes

        :param truncate: boolean for whether the stock data array is truncated to a length such that
            ``len(self.daily_stock_data) % self.sequence_length = 0``.
        """
        self.daily_stock_data = self.coi.return_numpy_array_of_company_daily_stock_percent_change()
        if truncate:
            mod = len(self.daily_stock_data) % self.sequence_segment_length
            if mod != 0:
                self.daily_stock_data = self.daily_stock_data[:-mod]

        try:
            assert len(self.daily_stock_data) >= 2 * self.sequence_segment_length
        except AssertionError:
            print(BColors.FAIL + "The specified segment length for the data to be split up into, {}, would result in "
                                 "a dataset of only one segment because the self.daily_stock_data array is of length {}"
                                 "; a minimum of 2 must be created for a train/test split (although there clearly needs"
                                 " to be more than 2 data points to train the model)."
                  .format(self.sequence_segment_length, len(self.daily_stock_data)) + BColors.WHITE)
            raise AssertionError

    def populate_test_train(self, rand_seed: int = -1):
        r"""
        Populates ``self.train_data`` and ``self.test_data`` tensors with complimentary subsets of the sequences of
        ``self.daily_stock_data``, where the sequences are the ``self.sequence_length`` length sequences of data that,
        when  concatenated, comprise ``self.daily_stock_data``.

        :param rand_seed: value to seed the random number generator; if -1 (or any value < 0), then do not
            seed the random number generator.
        """
        num_segments = len(self.daily_stock_data) // self.sequence_segment_length  # floor divide is used to return an
        # integer (should be no rounding)

        segmented_data = self.daily_stock_data.reshape(num_segments, self.sequence_segment_length)
        num_train_segments = round(num_segments * self.train_data_prop)
        if num_segments == num_train_segments:
            # If true, this means that there would be no data for testing (because the train/test ratio is very high
            # and/or there is too little data given self.sequence_segment_length
            num_train_segments -= 1

        if rand_seed >= 0:
            np.random.seed(rand_seed)

        all_indices = np.array(range(num_segments), dtype=np.int64)
        np.random.shuffle(all_indices)
        self.train_sample_indices = all_indices[0:num_train_segments]
        self.test_sample_indices = np.array(list(set(range(num_segments)) - set(self.train_sample_indices)))
        del all_indices

        # None indicates an empty dimension for the channels of the data (of which there is 1)
        train_segments = torch.from_numpy(segmented_data[self.train_sample_indices][:, None, :]).float()
        test_segments = torch.from_numpy(segmented_data[self.test_sample_indices][:, None, :]).float()
        del segmented_data

        X_train: Tensor = train_segments
        y_train: Tensor = train_segments[:, :, train_segments.shape[2] - self.label_length:]
        X_test: Tensor = test_segments
        y_test: Tensor = test_segments[:, :, test_segments.shape[2] - self.label_length:]
        self.train_set = TensorDataset(X_train, y_train)
        self.test_set = TensorDataset(X_test, y_test)

    def return_loaders(self) -> [DataLoader, DataLoader]:
        r"""
        Returns the :ref:`torch.utils.data.Dataloader` objects for the training and test sets

        :return: training DataLoader
        :return: testing DataLoader
        """
        if self.__togpu_works__ == 1:
            return [
                DataLoader(
                    self.train_set,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True  # speeds up the host-to-device transfer
                ),
                DataLoader(
                    self.test_set,
                    batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True  # speeds up the host-to-device transfer
                )
            ]
        else:
            return [
                DataLoader(
                    self.train_set,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers
                ),
                DataLoader(
                    self.test_set,
                    batch_size=self.test_batch_size,
                    num_workers=self.num_workers
                )
            ]

    def populate_loaders(self):
        r"""
        Populates :attr:`train_loader`, :attr:`test_laoder`, :attr:`train_loader_len`, and `:attr:`test_loader_len`
        attributes.
        """
        self.train_loader, self.test_loader = self.return_loaders()
        self.train_loader_len = len(self.train_loader)
        self.test_loader_len = len(self.test_loader)

    def forward(self, x: torch.Tensor):
        r"""
        Completes a forward pass of data through the network. The tensor passed in is of shape (batch size, features,
        sequence length), and the output is of shape (batch size, 1, sequence length).
        TODO: add support for more than one feature while still outputting a single feature prediction
        The data is passed through a LSTM layer with an arbitrary number of layers and an arbitrary hidden size (as
        defined by :attr:`lstm_hidden_size` and :attr:`lstm_num_layers`

        :param x:
        :return:
        """
        x = x.permute(2, 0, 1)  # input x needs to be converted from (batch_size, features, seqence_length) to
        # (sequence_length, batch_size, features) before being passed through the LSTM
        lstm_out = torch.zeros((x.shape[0]), (x.shape[1]), 1)  # will store the output of the LSTM layer
        output, (h_n, c_n) = self.lstm.forward(x[0, None, :, :])  # pass in the first value of the sequence and let
        fc_output = self.fc_after_lstm.forward(output)  # reduces the hidden_size dimension to 1
        lstm_out[0, :, :] = fc_output[0, :, :]
        # Pytorch initialize the hidden layer; output is of shape (sequence_length, batch_size, features * hidden_size)
        # where, for now, the hidden_size = 1 and the sequence length = 1
        for x_ in range(1, x.shape[0]):  # loop over the rest of the sequence; pass in a value one at a time and save
            # the hidden state to pass to the next forward pass
            output, (h_n, c_n) = self.lstm.forward(x[x_, None, :, :], (h_n, c_n))
            fc_output = self.fc_after_lstm.forward(output)
            lstm_out[x_, :, :] = fc_output[0, :, :]
        # lstm_output is now of shape(sequence_length, batch_size, features * hidden_size); convert back to
        # (batch_size, features, sequence_length)
        lstm_out = lstm_out.permute(1, 2, 0)

        # run dropout on the output of the lstm
        # out = self.dropout(lstm_out)
        return lstm_out

    def do_training(self, num_epochs: int, verbose=True):
        """
        This method trains the network using data in :attr:`train_loader` and checks against the data in
        :attr:`test_loader` at the end of each epoch.. The forward pass through the network produces sequences of the
        same length as the input sequences. The sequences in the label data are of length :attr:`label_length`, so the
        output sequences are  cropped to length :attr:`label_length` before being passed through the MSE loss function.
        Because each element of the output sequence at position ``n`` is a prediction of the input element ``n+1``, the
        cropped windows of  the output sequences are given by the window that terminates at the second-to-last element
        of the output sequence.

        :param num_epochs: number of epochs to to run the training for
        :param verbose: if true, print diagnostic progress updates and final training and test loss
        """
        print("Train loader size:", self.train_loader_len)
        epoch_num = 0
        pass_num = 0
        training_start_time = time.time()
        train_loss_list = []
        train_loss_list_idx = []
        test_loss_list = []
        test_loss_list_idx = []

        if num_epochs <= 0:
            print(BColors.FAIL + "Cannot complete training with <= 0 specified epochs." + BColors.WHITE)
            raise Exception

        while epoch_num < num_epochs:
            if verbose:
                print("Epoch num: {} | Progress: ".format(epoch_num))
            pass_num_this_epoch = 0
            for i, data in enumerate(self.train_loader, 0):
                train_inputs, train_labels = data
                # send inputs and labels to the gpu if possible
                if self.__togpu_works__ == 1:
                    train_inputs.to(DEVICE)
                    train_labels.to(DEVICE)
                # otherwise, ``inputs`` and ``labels`` are already tensors

                self.optimizer.zero_grad()
                output = self.forward(train_inputs)
                train_loss_size = self.loss(output[:, :, output.shape[2] - self.label_length - 1:-1], train_labels)
                train_loss_size.backward()
                train_loss_list.append(train_loss_size.data.item())
                train_loss_list_idx.append(pass_num)
                self.optimizer.step()
                pass_num += 1
                pass_num_this_epoch += 1
                if verbose:
                    percent = round(100 * pass_num_this_epoch / self.train_loader_len)
                    percent_floored_by_10: int = (percent // 10)
                    end = " train loss size = {}".format(train_loss_list[-1])
                    if pass_num_this_epoch == self.train_loader_len:
                        end += "\n"
                    else:
                        end += "\r"
                    print(
                        "-> {}% [".format(percent) + "-" * percent_floored_by_10 + " " * (10 - percent_floored_by_10)
                        + "]", end=end)

            # do a run on the test set at the end of every epoch:
            test_loss_this_epoch = 0
            for i, data in enumerate(self.test_loader, 0):
                test_inputs, test_labels = data
                if self.__togpu_works__ == 1:  # send inputs and labels to the gpu if possible
                    test_inputs.to(DEVICE)
                    test_labels.to(DEVICE)
                output = self.forward(test_inputs)
                test_loss_size = self.loss(output[:, :, output.shape[2] - self.label_length - 1:-1], test_labels)
                test_loss_this_epoch += test_loss_size.data.item()
            test_loss_list.append(test_loss_this_epoch / len(self.test_loader))
            test_loss_list_idx.append(pass_num)
            epoch_num += 1
            if verbose:
                print("test loss size = {}".format(test_loss_list[-1]))

        if verbose:
            print("-----------------\n"
                  "Finished training\n"
                  " >         Duration: {}s\n"
                  " > Final train loss: {} (delta of {})\n"
                  " >  Final test loss: {} (delta of {})".format(round(time.time() - training_start_time, 4),
                                                                 round(train_loss_list[-1], 4),
                                                                 round(train_loss_list[-1] - train_loss_list[0], 4),
                                                                 round(test_loss_list[-1], 4),
                                                                 round(test_loss_list[-1] - test_loss_list[0], 4)))


if __name__ == "__main__":
    model: StockRNN
    model = StockRNN('IBM', 5, 5)
    model.peek_dataset()

    try:
        model.to(DEVICE)
        model.__togpu__(True)
    except RuntimeError:
        print(TO_GPU_FAIL_MSG)
        raise RuntimeError
    except AssertionError:
        print(TO_GPU_FAIL_MSG)
        model.__togpu__(False)

    model.do_training(10)
