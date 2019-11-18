"""
Code to train the RNN
(note: adapted from https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/)
TODO: remove the above 'adapted from...' note when the code becomes sufficiently different

@author: Duncan Mazza
"""

from warnings import warn
from torch import Tensor
from tests.BColors import BColors
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from src.get_data.pandas_stock_data import numpy_array_of_company_daily_stock_close_av, \
    numpy_array_of_company_daily_stock_close_yahoo
import matplotlib.pyplot as plt
import time

DEVICE = "cuda"  # selects the gpu to be used
TO_GPU_FAIL_MSG = BColors.FAIL + "Unable to successfully run model.to('{}'). If running in Collaboratory, make sure " \
                                 "that you have enabled the GPU your settings".format(DEVICE) + BColors.DEFAULT


class StockRNN(nn.Module):
    r"""
    Class for handling all RNN operations.
    """
    train_set: TensorDataset
    test_set: TensorDataset
    train_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, lstm_hidden_size: int, lstm_num_layers: int, ticker: str,
                 start_date: datetime = datetime(2017, 1, 1), end_date: datetime = datetime(2018, 1, 1),
                 sequence_segment_length: int = 10, drop_prob: float = 0.5, device: str = DEVICE,
                 auto_populate: bool = True, train_data_prop: float = 0.8, src: str = 'yahoo', lr: float = 1e-4,
                 train_batch_size: int = 2, test_batch_size: int = 2, num_workers: int = 2, label_length: int = 10):
        """
        TODO: documentation here

        :param lstm_input_size:
        :param lstm_hidden_size:
        :param lstm_num_layers:
        :param ticker:
        :param start_date:
        :param end_date:
        :param sequence_segment_length:
        :param drop_prob:
        :param device:
        :param auto_populate:
        :param train_data_prop:
        :param src:
        :param lr:
        :param train_batch_size:
        :param test_batch_size:
        :param num_workers:
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
        self.src = src
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_segment_length = sequence_segment_length
        self.auto_populate = auto_populate
        self.train_data_prop = train_data_prop
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.label_length = label_length

        # initialize objects used during forward pass
        self.lstm = nn.LSTM(self.sequence_segment_length, self.lstm_hidden_size, self.lstm_num_layers,
                            dropout=self.drop_prob)
        self.dropout = nn.Dropout(drop_prob)

        # initialize attributes with placeholder arrays
        self.daily_stock_data = np.array(0)
        self.train_sample_indices = np.array(0)
        self.test_sample_indices = np.array(0)

        # initialize optimizer and loss
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.auto_populate:
            self.populate_daily_stock_data()
            self.populate_test_train()
            self.populate_loaders()

    def __togpu__(self, succ):
        """
        Sets the value of :attr:`__togpu_works__`, which is used in such a way that expensive error catching isn't run
        every epoch of training.

        :param succ: boolean for whether ``.to(gpu)`` was called successfully
        """
        if succ:
            self.__togpu_works__ = 1
        else:
            self.__togpu_works__ = -1

    def return_loss_and_optimizer(self):
        """
        TODO: documentation here

        :return: :attr:`optimizer`
        :return: :attr:`loss`
        """
        return self.optimizer, self.loss

    def peek_dataset(self, figsize: (int, int) = (10, 5)):
        """
        Creates a simple line plot of the stock data

        TODO: add title and axis labels

        :param figsize: tuple of integers for :class:`plt.subplots` ``figsize`` argument
        """
        _, axs = plt.subplots(1, 1, figsize=figsize)
        axs.plot(self.daily_stock_data)
        plt.show()

    def forward(self, x: torch.tensor):
        r"""
        TODO: documentation here

        :param x:
        :param hx:

        :return: lstm_out
        :return: hx
        """
        # input x needs to be converted from (batch_size, features, seqence_length) to (sequence_length, batch_size,
        # features) before being passed through the LSTM
        # TODO: reshape the input before passing to the lstm
        lstm_out= self.lstm.forward(x)
        # lstm_out is of shape (sequence_length, batch_size, hidden_size)
        # TODO: reshape the output before returnign

        # run dropout on the output of the lstm
        # out = self.dropout(lstm_out)

        return lstm_out

    def populate_daily_stock_data(self, truncate: bool = True):
        r"""
        Populates ``self.daily_stock_data`` using relevant class attributes

        :param truncate: boolean for whether the stock data array is truncated to a length such that
            ``len(self.daily_stock_data) % self.sequence_length = 0``.
        """
        if self.src.__contains__('av-'):
            self.daily_stock_data = numpy_array_of_company_daily_stock_close_av(self.ticker, self.start_date,
                                                                                self.end_date)
        else:  # self.src == 'yahoo':
            self.daily_stock_data = numpy_array_of_company_daily_stock_close_yahoo(self.ticker, self.start_date,
                                                                                   self.end_date)
        if truncate:
            mod = len(self.daily_stock_data) % self.sequence_segment_length
            if mod != 0:
                self.daily_stock_data = self.daily_stock_data[:-mod]

        try:
            assert len(self.daily_stock_data) > 2 * self.sequence_segment_length
        except AssertionError:
            print(TO_GPU_FAIL_MSG)

    def populate_test_train(self, rand_seed: int = -1):
        r"""
        Populates ``self.train_data`` and ``self.test_data`` tensors with complimentary subsets of the sequences of
        ``self.daily_stock_data``, where the sequences are the ``self.sequence_length`` length sequences of data that,
        when  concatenated, comprise ``self.daily_stock_data``.

        :param rand_seed: value to seed the random number generator; if -1 (or any value < 0), then do not
            seed the random number generator.
        """
        num_segments = len(
            self.daily_stock_data) // self.sequence_segment_length  # floor divide is used to return an integer
        # (should be no rounding)

        segmented_data = self.daily_stock_data.reshape(num_segments, self.sequence_segment_length)
        num_train_segments = round(num_segments * self.train_data_prop)

        if rand_seed >= 0:
            np.random.seed(rand_seed)
        all_indices = np.array(range(num_segments), dtype=np.int64)
        np.random.shuffle(all_indices)
        self.train_sample_indices = all_indices[0:num_train_segments]
        self.test_sample_indices = np.array(list(set(range(num_segments)) - set(self.train_sample_indices)))
        del all_indices

        # None indicates an empty dimension for the channels of the data (of which there is 1)
        train_segments = torch.from_numpy(segmented_data[self.train_sample_indices][:, None, :])
        test_segments = torch.from_numpy(segmented_data[self.test_sample_indices][:, None, :])
        del segmented_data

        X_train = train_segments[:, :, :-1]
        y_train = train_segments[:, :, -self.label_length]
        X_test = test_segments[:, :, :-1]
        y_test = test_segments[:, :, -self.label_length]
        self.train_set = TensorDataset(X_train, y_train)
        self.test_set = TensorDataset(X_test, y_test)

    def return_loaders(self) -> [DataLoader, DataLoader]:
        """
        TODO: documentation here

        :return: training DataLoader
        :return: testing DataLoader
        """
        if self.__togpu_works__ == 1:
            return [
                DataLoader(
                    self.train_set,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True
                ),
                DataLoader(
                    self.test_set,
                    batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True
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
        """
        TODO: documentation here
        """
        [self.train_loader, self.test_loader] = self.return_loaders()

    def do_training(self, num_epochs: int):
        r"""
        TODO: documentation here
        """
        train_idx = 0
        epoch_num = 0
        training_start_time = time.time()

        while epoch_num < num_epochs:
            epoch_start_time = time.time()
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                print("Batch size: ", self.train_loader.batch_size)

                # send inputs and labels to the gpu if possible
                if self.__togpu_works__ == 1:
                    inputs.to(DEVICE)
                    labels.to(DEVICE)
                elif self.__togpu_works__ == 0:
                    try:
                        inputs.to(DEVICE)
                        labels.to(DEVICE)
                        model.__togpu__(True)
                    except RuntimeError:
                        print(TO_GPU_FAIL_MSG)
                        raise RuntimeError
                    except AssertionError:
                        print(TO_GPU_FAIL_MSG)
                        model.__togpu__(False)
                # otherwise, ``inputs`` and ``labels`` are already tensors

                # zero out the gradients before each pass
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                # loss = self.loss(outputs, labels)
                # loss.backward()
                # self.optimizer.step()


if __name__ == "__main__":
    model: StockRNN
    model = StockRNN(1, 1, 'IBM')  # TODO: update these parameters
    # model.peek_dataset()

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

# output_size = 1
# embedding_dim = 400
# hidden_dim = 512
# n_layers = 2
#
# model = StockRNN(output_size, embedding_dim, hidden_dim, n_layers)
# model.to(DEVICE)  # send to GPU
#
# lr = 0.005
# criterion = nn.BCELoss()  # binary cross-entropy loss between target and output
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
# epochs = 2
# counter = 0
# print_every = 1000
# clip = 5
# valid_loss_min = np.Inf
#
# model.train()
# for i in range(epochs):
#     h = model.init_hidden(batch_size)
#
#     for inputs, labels in train_loader:
#         counter += 1
#         h = tuple([e.data for e in h])
#         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#         model.zero_grad()
#         output, h = model(inputs, h)
#         loss = criterion(output.squeeze(), labels.float())
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#
#         if counter % print_every == 0:
#             val_h = model.init_hidden(batch_size)
#             val_losses = []
#             model.eval()
#             for inp, lab in val_loader:
#                 val_h = tuple([each.data for each in val_h])
#                 inp, lab = inp.to(DEVICE), lab.to(DEVICE)
#                 out, val_h = model(inp, val_h)
#                 val_loss = criterion(out.squeeze(), lab.float())
#                 val_losses.append(val_loss.item())
#
#             model.train()
#             print("Epoch: {}/{}...".format(i + 1, epochs),
#                   "Step: {}...".format(counter),
#                   "Loss: {:.6f}...".format(loss.item()),
#                   "Val Loss: {:.6f}".format(np.mean(val_losses)))
#             if np.mean(val_losses) <= valid_loss_min:
#                 torch.save(model.state_dict(), './state_dict.pt')
#                 print('Validation loss decreased ({:.6f} --> {:.6f}).'
#                       'Saving model ...'.format(valid_loss_min, np.mean(val_losses)))
#                 valid_loss_min = np.mean(val_losses)