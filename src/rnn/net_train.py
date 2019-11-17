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
import numpy as np
from datetime import datetime
from src.get_data.pandas_stock_data import numpy_array_of_company_daily_stock_close_av, \
    numpy_array_of_company_daily_stock_close_yahoo
import matplotlib.pyplot as plt
import matplotlib

DEVICE = "cuda"  # selects the gpu to be used


class StockRNN(nn.Module):
    r"""
    Class for handling all RNN operations.
    """
    train_data: Tensor
    test_data: Tensor

    def __init__(self, lstm_input_size: int, lstm_output_size: int, lstm_num_layers: int, ticker: str,
                 start_date: datetime = datetime(2017, 1, 1), end_date: datetime = datetime(2018, 1, 1),
                 sequence_length: int = 10, drop_prob: float = 0.5, device: str = DEVICE, auto_populate: bool = True,
                 train_data_prop: float = 0.8, src: str = 'yahoo', lr: float = 1e-4, train_batch_size: int = 30,
                 test_batch_size: int = 30, num_workers: int = 2):
        r"""
        TODO: documentation here
        """
        super(StockRNN, self).__init__()

        # __init__ params
        self.lstm_input_size = lstm_input_size
        self.lstm_output_size = lstm_output_size
        self.lstm_num_layers = lstm_num_layers
        self.drop_prob = drop_prob
        self.device = device
        self.ticker = ticker
        self.src = src
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.auto_populate = auto_populate
        self.train_data_prop = train_data_prop
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        # initialize objects used during forward pass
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_output_size, self.lstm_num_layers, dropout=self.drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # self.conv1 = nn.Conv1d()
        # self.sigmoid = nn.Sigmoid()

        # initialize attributes with placeholder values
        self.daily_stock_data = np.array(0)
        self.test_data = torch.tensor(0)
        self.train_data = torch.tensor(0)
        self.test_sample_indices = np.array(0)
        self.train_sample_indices = np.array(0)

        # initialize optimizer and loss
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # initialize samplers
        self.train_sampler = SubsetRandomSampler(np.arange(1, dtype=np.int64))
        self.test_sampler = SubsetRandomSampler(np.arange(1, dtype=np.int64))

        if self.auto_populate:
            self.populate_daily_stock_data()
            self.populate_test_train()

    def return_loss_and_optimizer(self):
        """
        TODO: documentation here

        :return: :attr:`optimizer`
        :return: :attr:`loss`
        """
        return self.optimizer, self.loss

    def return_loaders(self):
        """
        TODO: documentation here

        :return: training DataLoader
        :return: testing DataLoader
        """
        return [
            DataLoader(
                self.train_data,
                batch_size=self.train_batch_size,
                sampler=self.train_sampler,
                num_workers=self.num_workers
            ),
            DataLoader(
                self.test_data,
                batch_size=self.test_batch_size,
                sampler=self.test_sampler,
                num_workers=self.num_workers
            )
        ]

    def peek_dataset(self, figsize: (int, int) = (10, 5)):
        """
        Creates a simple line plot of the stock data

        TODO: add title and axis labels

        :param figsize: tuple of integers for :class:`plt.subplots` ``figsize`` argument
        """
        _, axs = plt.subplots(1, 1, figsize=figsize)
        axs.plot(self.daily_stock_data)
        plt.show()

    def forward(self, x, hidden):
        """
        TODO: documentation here
        """
        batch_size = x.size(0)
        x = x.long()  # converts to 64 bit integer
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # pass the lstm output through the fully connected layer
        # out = self.dropout(lstm_out)
        # out = self.fc(out)
        # out = self.sigmoid(out)

        # out = out.view(batch_size, -1)
        # out = out[:, -1]
        # return out, hidden
        return lstm_out, hidden

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        hidden = []
        for _ in range(self.lstm_num_layers):
            hidden.append(weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden

    def populate_daily_stock_data(self, truncate: bool = True):
        """
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
            mod = len(self.daily_stock_data) % self.sequence_length
            if mod != 0:
                self.daily_stock_data = self.daily_stock_data[:-mod]

        try:
            assert len(self.daily_stock_data) > 2 * self.sequence_length
        except AssertionError:
            print(BColors.FAIL + "Not enough data was acquired; at least 2*self.sequence_length data points must be "
                                 "collected, but {} data points were collected".format(len(self.daily_stock_data)))

    def populate_test_train(self, rand_seed: int = -1):
        """
        Populates ``self.train_data`` and ``self.test_data`` tensors with complimentary subsets of the sequences of
        ``self.daily_stock_data``, where the sequences are the ``self.sequence_length`` length sequences of data that,
        when  concatenated, comprise ``self.daily_stock_data``.

        :param rand_seed: value to seed the random number generator; if -1 (or any value < 0), then do not
            seed the random number generator.
        """
        num_segments = len(self.daily_stock_data) // self.sequence_length  # floor divide is used to return an integer
        # (should be no rounding)

        segmented_data = self.daily_stock_data.reshape(num_segments, self.sequence_length)
        num_train_segments = round(num_segments * self.train_data_prop)

        if rand_seed >= 0:
            np.random.seed(rand_seed)
        all_indices = np.array(range(num_segments), dtype=np.int64)
        np.random.shuffle(all_indices)
        self.train_sample_indices = all_indices[0:num_train_segments]
        self.test_sample_indices = np.array(list(set(range(num_segments)) - set(self.train_sample_indices)))
        self.train_data = torch.from_numpy(segmented_data[self.train_sample_indices])
        self.test_data = torch.from_numpy(segmented_data[self.test_sample_indices])


if __name__ == "__main__":
    model: StockRNN
    model = StockRNN(1, 1, 1, 'IBM', auto_populate=True, src='yahoo')  # TODO: update these parameters
    model.peek_dataset()
    try:
        model.to(DEVICE)
    except RuntimeError:
        print(BColors.FAIL + "Unable to successfully run model.to('{}'). If running in Collaboratory, make sure that "
                             "you have enabled the GPU your settings".format(DEVICE))
        raise RuntimeError
    except AssertionError:
        print(BColors.WARNING + "Unable to successfully run model.to('{}'). If running in Collaboratory, make sure that "
                             "you have enabled the GPU your settings".format(DEVICE))

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
