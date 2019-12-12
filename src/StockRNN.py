"""
Code to train the RNN

@author: Duncan Mazza
"""

from torch import Tensor
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from datetime import datetime
from src.get_data import Company
import matplotlib.pyplot as plt
import time
from pandas import _libs
from pandas_datareader._utils import RemoteDataError
import os

ZERO_TIME = " 00:00:00"

DEVICE = "cuda"  # selects the gpu to be used
TO_GPU_FAIL_MSG = "Unable to successfully run model.to('{}'). If running in Collaboratory, make sure " \
                  "that you have enabled the GPU your settings".format(DEVICE)


class Rescaler:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.train_max = 0
        self.train_min = 0

    def rescale_train(self, train_data):
        self.train_max = np.max(train_data)
        self.train_min = np.min(train_data)
        return (self.max_val - self.min_val) * (train_data - self.train_min) / (self.train_max - self.train_min) + \
               self.min_val

    def rescale_test(self, test_data):
        return (self.max_val - self.min_val) * (test_data - self.train_min) / (self.train_max - self.train_min) + \
               self.min_val


class StockRNN(nn.Module):
    r"""
    Class for handling all RNN operations.
    """
    train_set: TensorDataset
    test_set: TensorDataset
    train_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, ticker: str, lstm_hidden_size: int = 100, lstm_num_layers: int = 2, to_compare: [str, ] = None,
                 start_date: datetime = datetime(2017, 1, 1), end_date: datetime = datetime(2018, 1, 1),
                 sequence_segment_length: int = 50, drop_prob: float = 0.5, device: str = DEVICE,
                 auto_populate: bool = True, train_data_prop: float = 0.8, lr: float = 1e-4,
                 train_batch_size: int = 10, test_batch_size: int = 4, num_workers: int = 2, label_length: int = 30,
                 try_load_weights: bool = False, save_state_dict: bool = True, rolling_avg_length: int = 10):
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
        self.save_state_dict = save_state_dict
        self.rolling_avg_length = rolling_avg_length

        if label_length >= self.sequence_segment_length:
            print("Label length was specified to be {}, but cannot be >= self.sequence_segment_length; setting "
                  "self.label_length to self.sequence_segment_length - 1.")
            self.label_length = self.sequence_segment_length - 1
        else:
            self.label_length = label_length

        # company in index 0 is the company whose stock is being predicted
        self.companies = [Company(self.ticker, self.start_date, self.end_date)]

        start_date_changes = []
        end_date_changes = []
        if to_compare is not None:
            to_compare.sort()
            for company_ticker in to_compare:
                try:
                    self.companies.append(Company(company_ticker, self.start_date, self.end_date))
                except KeyError:
                    print("There was a KeyError exception raised when accessing data for the ticker {}; will skip this "
                          "ticker".format(company_ticker))
                    continue
                except _libs.tslibs.np_datetime.OutOfBoundsDatetime:
                    print("There was a _libs.tslibs.np_datetime.OutOfBoundsDatetime exception raised when accessing "
                          "data for the ticker {}; will skip this ticker".format(company_ticker))
                    continue
                except RemoteDataError:
                    print("There was a RemoteDataError when fetching data for ticker '{}'; will skip this ticker"
                          .format(company_ticker))
                    continue

                if self.companies[-1].start_date_changed:
                    start_date_changes.append(self.companies[-1].start_date)
                if self.companies[-1].end_date_changed:
                    end_date_changes.append(self.companies[-1].end_date)

        self.num_companies = len(self.companies)

        if len(start_date_changes) != 0:  # revise the start date of all of the data if necessary
            self.start_date = max(start_date_changes)
            for company in self.companies:
                company.revise_start_date(self.start_date)
            print("Data did not exist for every ticker at start date of {}; revising to the most recent starting time "
                  "(common among all companies' data) of {}".format(start_date.__str__().strip(ZERO_TIME),
                                                                    self.start_date.__str__().strip(ZERO_TIME)))
        # revise the end date of all of the data
        if len(end_date_changes) != 0:
            self.end_date = min(end_date_changes)
            for company in self.companies:
                company.revise_end_date(self.end_date)
            print("Data did not exist for every ticker at end date of {}; revising to the earliest ending time "
                  "(common among all companies' data) of {}".format(end_date.__str__().strip(ZERO_TIME),
                                                                    self.end_date.__str__().strip(ZERO_TIME)))
        self.start_date_str = self.start_date.__str__().strip(ZERO_TIME)
        self.end_date_str = self.end_date.__str__().strip(ZERO_TIME)

        # sting that describes the parameters for this model such that files for weights can be successfully loaded
        if self.num_companies > 1:
            considering_string = "_CONSIDERING_" + "&".join(list(map(lambda company:
                                                                     company.ticker, self.companies[1:])))
        else:
            considering_string = ""
        self.identifier = "MODEL_FOR_" + self.companies[0].ticker + considering_string + \
                          "_WITH_lstm_hidden_size_{}_lstm_num_layers_{}_input_size_{}_rolling_avg_length_{}_sequence_" \
                          "segment_length_{}".format(
                              self.lstm_hidden_size,
                              self.lstm_num_layers,
                              self.num_companies,
                              self.rolling_avg_length,
                              self.sequence_segment_length)

        self.model_weights_path = os.path.join(os.getcwd(), ".cache", self.identifier + ".bin")

        # initialize objects used during forward pass
        self.lstm = nn.LSTM(input_size=self.num_companies, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, dropout=self.drop_prob, batch_first=True)
        self.post_lstm_dropout = nn.Dropout(p=self.drop_prob)
        self.fc_1 = nn.Linear(self.lstm_hidden_size, 10)
        self.fc_2 = nn.Linear(10, self.num_companies)
        self.tanh = nn.Tanh()
        self.rescaler = Rescaler(-0.5, 0.5)

        # initialize attributes with placeholder arrays
        self.daily_stock_data = np.array(0)
        self.train_sample_indices = np.array(0)
        self.test_sample_indices = np.array(0)
        self.train_loader_len = 0
        self.test_loader_len = 0
        self.data_len = 0

        # initialize optimizer and loss
        self.loss = nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.auto_populate:
            self.populate_daily_stock_data()
            self.populate_test_train()
            self.populate_loaders()

        if try_load_weights:
            try:
                weights = torch.load(self.model_weights_path)
                self.load_state_dict(weights)
                print("Loded weights from file")
            except FileNotFoundError:
                print("Tried loading state dict from file but could not find cached file")
            except:
                print("WARNING: Could not load state dict for an unknown reason")

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
        if self.num_companies == 1:
            _, axes = plt.subplots(1, 1, figsize=figsize)
            axes.plot(self.daily_stock_data[0, :])
            axes.set_title("'{}' closing price day-over-day % change from {} to {}".format(self.companies[0].ticker,
                                                                                           self.start_date_str,
                                                                                           self.end_date_str))
            axes.set_xlabel("Time")
            axes.set_ylabel("Price (USD)")
        else:
            _, axes = plt.subplots(2, 1, figsize=figsize)
            axes[0].plot(self.daily_stock_data[0, :])
            axes[0].set_title(
                "'{}' closing price day-over-day % change from {} to {}".format(self.companies[0].ticker,
                                                                                self.start_date_str, self.end_date_str))
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel("Price (USD)")

            for c, company in enumerate(self.companies):
                axes[1].plot(self.daily_stock_data[c, :], label=company.ticker)
            axes[1].legend()
            axes[1].set_title(
                "All companies' closing price day-over-day % change from {} to {}".format(self.start_date_str,
                                                                                          self.end_date_str))
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Price (USD)")

        plt.show()

    def populate_daily_stock_data(self):
        r"""
        Populates ``self.daily_stock_data`` with the day-over-day percent change of the closing stock prices. The data
        for each company is truncated such that each company's array of data is the same length as the rest and such
        that their length is divisible by :attr:` sequence_segment_length`.
        """
        daily_stock_data = []
        daily_stock_data_lens = []
        data_is_of_same_len = True
        for company in self.companies:
            daily_stock_data.append(company.return_numpy_array_of_company_daily_stock_percent_change(
                rolling_avg_length=self.rolling_avg_length))
            daily_stock_data_lens.append(len(daily_stock_data[-1]))
            if daily_stock_data_lens[0] != daily_stock_data_lens[-1]:
                data_is_of_same_len = False

        self.data_len = min(daily_stock_data_lens)
        mod = self.data_len % self.sequence_segment_length
        if not data_is_of_same_len or mod != 0:
            self.data_len -= mod
            for c in range(self.num_companies):
                daily_stock_data[c] = daily_stock_data[c][-self.data_len:]

        try:
            assert self.data_len >= 2 * self.sequence_segment_length
        except AssertionError:
            print("The specified segment length for the data to be split up into, {}, would result in "
                  "a dataset of only one segment because the self.daily_stock_data array is of length {}"
                  "; a minimum of 2 must be created for a train/test split (although there clearly needs"
                  " to be more than 2 data points to train the model)."
                  .format(self.sequence_segment_length, self.data_len))
            raise AssertionError

        self.daily_stock_data = np.array(daily_stock_data)

    def populate_test_train(self, rand_seed: int = -1):
        r"""
        Populates ``self.train_data`` and ``self.test_data`` tensors with complimentary subsets of the sequences of
        ``self.daily_stock_data``, where the sequences are the ``self.sequence_length`` length sequences of data that,
        when  concatenated, comprise ``self.daily_stock_data``.

        :param rand_seed: value to seed the random number generator; if -1 (or any value < 0), then do not
            seed the random number generator.
        """
        num_segments = self.data_len // self.sequence_segment_length  # floor divide is used to return an
        # integer (should be no rounding)

        # shape of segmented_data: (batch_size, num_features, sequence_length)
        segmented_data = np.zeros((num_segments, self.num_companies, self.sequence_segment_length))
        for c in range(self.num_companies):
            segmented_data[:, c, :] = self.daily_stock_data[c, :].reshape((num_segments, self.sequence_segment_length))
        num_train_segments = round(num_segments * self.train_data_prop)
        if num_segments == num_train_segments:
            # If true, this means that there would be no data for testing (because the train/test ratio is very high
            # and/or there is too little data given self.sequence_segment_length
            num_train_segments -= 1

        if rand_seed >= 0:
            np.random.seed(rand_seed)  # useful for unit testing

        all_indices = np.array(range(num_segments), dtype=np.int64)
        np.random.shuffle(all_indices)
        self.train_sample_indices = all_indices[0:num_train_segments]
        self.test_sample_indices = np.array(list(set(range(num_segments)) - set(self.train_sample_indices)))
        del all_indices

        # X_train: Tensor = torch.from_numpy(self.rescaler.rescale_train(segmented_data[self.train_sample_indices, :, :])).float()
        # X_test: Tensor = torch.from_numpy(self.rescaler.rescale_test(segmented_data[self.test_sample_indices, :, :])).float()
        X_train: Tensor = torch.from_numpy(segmented_data[self.train_sample_indices, :, :]).float()
        X_test: Tensor = torch.from_numpy(segmented_data[self.test_sample_indices, :, :]).float()
        del segmented_data
        # the data for the labels is the data in the first position of the features dimension
        y_train: Tensor = X_train[:, :, -self.label_length:]
        y_test: Tensor = X_test[:, :, -self.label_length:]
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

    def forward(self, X: torch.Tensor, predict_beyond: int = 0):
        r"""
        Completes a forward pass of data through the network. The tensor passed in is of shape (batch size, features,
        sequence length), and the output is of shape (batch size, 1, sequence length). The data is passed through a LSTM
        layer with an arbitrary number of layers and an arbitrary hidden size (as defined by :attr:`lstm_hidden_size`
        and :attr:`lstm_num_layers`

        :param X: input matrix of data of shape: (batch size, features (number of companies), sequence length)
        :param predict_beyond: TODO
        :return: output of the forward pass of the data through the network (same shape as input)
        """
        X = X.permute(0, 2, 1)  # input x needs to be converted from (batch_size, features, sequence_length) to
        # (batch_size, sequence_length, features)

        output, (h, c) = self.lstm.forward(X)
        output = self.post_lstm_dropout(output)  # dropout built into LSTM object doesnt work on last layer of LSTM
        output = self.fc_1.forward(output)
        output = self.tanh(output)
        output = self.fc_2.forward(output)
        output = self.tanh(output)

        if predict_beyond == 0:
            output = output.permute(0, 2, 1)
            return output
        else:
            new_output = torch.zeros(output.shape[0], self.sequence_segment_length - 1 + predict_beyond,
                                     output.shape[2])
            new_output[:, :output.shape[1], :] = output
            for i in range(predict_beyond):
                predict_beyond_out, (h, c) = self.lstm.forward(output[:, -1, None, :], (h, c))
                predict_beyond_out = self.fc_1.forward(predict_beyond_out)
                predict_beyond_out = self.tanh(predict_beyond_out)
                predict_beyond_out = self.fc_2.forward(predict_beyond_out)
                predict_beyond_out = self.tanh(predict_beyond_out)
                new_output[:, self.sequence_segment_length - 1 + i, None, :] = predict_beyond_out
            new_output = new_output.permute(0, 2, 1)
            return new_output

    def do_training(self, num_epochs: int, verbose=True, plot_output: bool = True,
                    plot_output_figsize: (int, int) = (5, 10), plot_loss: bool = True,
                    plot_loss_figsize: (int, int) = (7, 5)):
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
        :param plot_output: if true, plot the results of the final pass through the LSTM with a randomly selected
        segment of data
        :param plot_output_figsize: ``figsize`` argument for the output plot
        :param plot_loss: if true, plot the training and test loss
        :param plot_loss_figsize: ``figsize`` argument for the loss plot
        """
        # self.train(True)
        epoch_num = 0
        pass_num = 0
        training_start_time = time.time()
        train_loss_list = []
        train_loss_list_idx = []
        test_loss_list = []
        test_loss_list_idx = []

        if num_epochs <= 0:
            print("Specified number of epochs is <= 0; it must be > 0, so it is set to 1.")
            num_epochs = 1

        while epoch_num <= num_epochs:
            if verbose:
                print("Epoch num: {}/{}: ".format(epoch_num, num_epochs))
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
                if verbose:
                    percent = round(100 * (i + 1) / self.train_loader_len)
                    percent_floored_by_10: int = (percent // 10)
                    front = '\r' if i != 0 else ""
                    print(
                        "{} > {}% [".format(front, percent) + "-" * percent_floored_by_10 + " " * (
                                10 - percent_floored_by_10)
                        + "] train loss size = {}".format(round(train_loss_list[-1], 4)), end="")

            # do a run on the test set at the end of every epoch:
            test_loss_this_epoch = 0
            subplot_val = 0
            if epoch_num == num_epochs and plot_output:
                subplot_val = self.test_loader_len if self.test_loader_len <= 3 else 3
                _, axes = plt.subplots(subplot_val, 1, figsize=plot_output_figsize)
                if subplot_val == 1:
                    axes = [axes, ]
            for i, data in enumerate(self.test_loader, 0):
                test_inputs, test_labels = data
                if self.__togpu_works__ == 1:  # send inputs and labels to the gpu if possible
                    test_inputs.to(DEVICE)
                    test_labels.to(DEVICE)
                output = self.forward(test_inputs)
                test_loss_size = self.loss(output[:, :, output.shape[2] - self.label_length - 1:-1], test_labels)
                test_loss_this_epoch += test_loss_size.data.item()

                if epoch_num == num_epochs and plot_output and i < subplot_val:
                    axes[i].plot(np.arange(0, self.sequence_segment_length, 1), test_inputs[0, 0, :].detach().numpy(),
                                 label="orig")
                    axes[i].plot(np.arange(1, self.sequence_segment_length + 1, 1), output[0, 0, :].detach().numpy(),
                                 label="pred")
                    axes[i].set_title(
                        "'{}' closing price day-over-day % change\nfrom {} to {}: Example {} of\nOriginal vs. Model "
                        "Output".format(self.companies[0].ticker, self.start_date_str, self.end_date_str, i))
                    axes[i].set_xlabel("Time")
                    axes[i].set_ylabel("% Change of Stock (USD)")

            if epoch_num == num_epochs and plot_output:
                plt.legend()
                plt.show()

            test_loss_list.append(test_loss_this_epoch / len(self.test_loader))
            test_loss_list_idx.append(pass_num)
            epoch_num += 1
            if verbose:
                print(" | test loss size = {}".format(round(test_loss_list[-1], 4)))

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

        if self.save_state_dict:
            if not os.path.isdir(os.path.join(os.getcwd(), ".cache")):
                os.mkdir(os.path.join(os.getcwd(), ".cache"))
            try:
                torch.save(self.state_dict(), self.model_weights_path)
                print(" > (saved model weights to '{}' folder)".format(os.path.join(os.getcwd(), ".cache")))
            except:
                print("WARNING: an unknown exception occured when trying to save model weights")

        if plot_loss:
            _, axes = plt.subplots(1, 1, figsize=plot_loss_figsize)
            axes.plot(train_loss_list_idx, train_loss_list, label="train")
            axes.plot(test_loss_list_idx, test_loss_list, label="test")
            axes.set_xlabel("Train data forward pass index")
            axes.set_ylabel("Loss magnitude")
            axes.set_title("Train and Testing Data Loss over Training Duration")
            plt.legend()
            plt.show()

    def make_prediction_with_validation(self, predict_beyond: int = 30, num_plots: int = 2, plt_scl=8):
        r"""
        Randomly selects data from the dataset and makes a prediction ``predict_beyond`` days out. The sequence length
        of the data passed to the forward pass is given by ``self.sequence_segment_length - predict_beyond``, and the
        actual values of the stock are shown alongside.

        :param predict_beyond: days to predict ahead in the future
        :param num_plots:
        :return:
        """
        # self.eval()  # equivalent to calling self.train(False)
        forward_seq_len = self.sequence_segment_length + predict_beyond
        data_start_indices = np.random.choice(self.daily_stock_data.shape[1] - forward_seq_len, num_plots)

        start_dates = []
        end_dates = []
        pred_data_start_indicies = []
        make_pred_data = torch.zeros((num_plots, self.num_companies, forward_seq_len))
        for i in range(num_plots):
            make_pred_data[i, :, :] = torch.from_numpy(
                self.daily_stock_data[:, data_start_indices[i]:data_start_indices[i] + forward_seq_len])
            start_dates.append(self.companies[0].get_date_at_index(data_start_indices[i]))
            end_dates.append(self.companies[0].get_date_at_index(data_start_indices[i] + forward_seq_len))
            pred_data_start_indicies.append(data_start_indices[i] + forward_seq_len - predict_beyond)
        output = self.forward(make_pred_data[:, :, :-(predict_beyond + 1)], predict_beyond)
        output_numpy = output.detach().numpy()
        _, axes = plt.subplots(num_plots, 3, figsize=(plt_scl, plt_scl))
        for ax in range(num_plots):
            axes[ax][0].plot(np.arange(self.sequence_segment_length, forward_seq_len),
                             output_numpy[ax, 0, -predict_beyond:], color='green', label="pred. beyond",
                             linestyle='--', marker='o')
            axes[ax][0].plot(np.arange(1, self.sequence_segment_length),
                             output_numpy[ax, 0, :self.sequence_segment_length - 1], color='gray',
                             label="pred. over input", linestyle='--', marker='o')
            axes[ax][0].plot(np.arange(0, self.sequence_segment_length),
                             make_pred_data.detach().numpy()[ax, 0, :self.sequence_segment_length], color='red',
                             label="input", linestyle='-', marker='o')
            axes[ax][0].plot(np.arange(self.sequence_segment_length, forward_seq_len),
                             make_pred_data.detach().numpy()[ax, 0, self.sequence_segment_length:], label="actual",
                             linestyle='-', marker='o')

            axes[ax][0].set_title("{} % change from\n{} to {}".format(self.companies[0].ticker,
                                                                      start_dates[ax], end_dates[ax]))
            axes[ax][0].set_xlabel("Business days since {}".format(start_dates[ax]))
            axes[ax][0].set_ylabel("% Change")
            axes[ax][0].legend()

            pred_stock = self.companies[0].reconstruct_stock_from_percent_change(output_numpy[ax, 0, -predict_beyond:],
                                                                                 initial_condition_index=(
                                                                                             pred_data_start_indicies[
                                                                                                 ax] - 1))[1:]
            axes[ax][1].plot(np.arange(self.sequence_segment_length, forward_seq_len), pred_stock, color="green",
                             label="pred. beyond", linestyle='--', marker='o')
            orig_stock = self.companies[0].data_frame["Close"].iloc[
                list(range(data_start_indices[ax], pred_data_start_indicies[ax], 1))]
            axes[ax][1].plot(np.arange(0, self.sequence_segment_length), orig_stock, color='red', label="input",
                             linestyle='-', marker='o')
            actual_stock = self.companies[0].data_frame["Close"].iloc[
                list(range(pred_data_start_indicies[ax], pred_data_start_indicies[ax] + predict_beyond))]
            axes[ax][1].plot(np.arange(self.sequence_segment_length, forward_seq_len), actual_stock, label="actual",
                             linestyle='-', marker='o')
            axes[ax][1].set_title("{} stock from\n{} to {}".format(self.companies[0].ticker, start_dates[ax], end_dates[ax]))
            axes[ax][1].set_xlabel("Business days since {}".format(start_dates[ax]))
            axes[ax][1].set_ylabel("$")
            axes[ax][1].legend()

            axes[ax][2].plot(np.arange(1, predict_beyond+1), np.abs(pred_stock - actual_stock), label="disparity",
                             linestyle="", marker="o")
            axes[ax][2].set_title("Disparity of Predicted and Actual Stock")
            axes[ax][2].set_xlabel("Num. predicted days out {}".format(start_dates[ax]))
            axes[ax][2].set_ylabel("Absolute difference between prediction and reality")

        plt.show()


if __name__ == "__main__":
    model: StockRNN

    # set to switch between loading saved weights if available
    try_load_weights = True

    model = StockRNN("IBM", to_compare=["HPE", "XRX", "ACN", "ORCL"], start_date=datetime(2012, 1, 1),
                     end_date=datetime(2019, 1, 1), try_load_weights=try_load_weights)
    # model = StockRNN("dummy")
    # model.peek_dataset()

    try:
        model.to(DEVICE)
        model.__togpu__(True)
    except RuntimeError:
        print(TO_GPU_FAIL_MSG)
    except AssertionError:
        print(TO_GPU_FAIL_MSG)
        model.__togpu__(False)

    model.do_training(num_epochs=100)

    model.make_prediction_with_validation()
