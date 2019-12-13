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


class StockRNN(nn.Module):
    r"""
    Class for training on and predicting stocks using a LSTM network
    """
    train_set: TensorDataset
    test_set: TensorDataset
    train_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, ticker: str, lstm_hidden_size: int = 100, lstm_num_layers: int = 2, to_compare: [str, ] = None,
                 train_start_date: datetime = datetime(2017, 1, 1), train_end_date: datetime = datetime(2018, 1, 1),
                 sequence_segment_length: int = 50, drop_prob: float = 0.3, device: str = DEVICE,
                 auto_populate: bool = True, train_data_prop: float = 0.8, lr: float = 1e-4,
                 train_batch_size: int = 10, test_batch_size: int = 4, num_workers: int = 2, label_length: int = 30,
                 try_load_weights: bool = False, save_state_dict: bool = True):
        r"""
        :param lstm_hidden_size: size of the lstm hidden layer
        :param lstm_num_layers: number of layers for the lstm
        :param ticker: ticker of company whose stock you want to predict
        :param to_compare: ticker of companies whose stock will be part of the features of the dataset
        :param train_start_date: date to request data from
        :param train_end_date: date to request data to
        :param sequence_segment_length: length of sequences to train the model on
        :param drop_prob: probability for dropout layers
        :param device: string for device to try sending the tensors to (i.e. "cuda")
        :param auto_populate: automatically calls all 'populate' functions in the constructor
        :param train_data_prop: proportion of data set to allocate to training data
        :param lr: learning rate for the optimizer
        :param train_batch_size: batch size for the training data
        :param test_batch_size:batch size for the testing data
        :param num_workers: parameter for Pytorch DataLoaders
        :param label_length: length of data (starting at the end of each sequence segment) to consider for the loss
        :param try_load_weights: boolean for whether the model should search for a cached model state dictionary
        :param save_state_dict: boolean for whether the model should cache its weights as a state dictionary
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
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.sequence_segment_length = sequence_segment_length
        self.auto_populate = auto_populate
        self.train_data_prop = train_data_prop
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.save_state_dict = save_state_dict

        if label_length >= self.sequence_segment_length:
            print("Label length was specified to be {}, but cannot be >= self.sequence_segment_length; setting "
                  "self.label_length to self.sequence_segment_length - 1.")
            self.label_length = self.sequence_segment_length - 1
        else:
            self.label_length = label_length

        # company in index 0 is the company whose stock is being predicted
        self.companies = [Company(self.ticker, self.train_start_date, self.train_end_date)]

        start_date_changes = []
        end_date_changes = []
        if to_compare is not None:
            to_compare.sort()
            for company_ticker in to_compare:
                try:
                    self.companies.append(Company(company_ticker, self.train_start_date, self.train_end_date))
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
            self.train_start_date = max(start_date_changes)
            for company in self.companies:
                company.revise_start_date(self.train_start_date)
            print("Data did not exist for every ticker at start date of {}; revising to the most recent starting time "
                  "(common among all companies' data) of {}".format(train_start_date.__str__().strip(ZERO_TIME),
                                                                    self.train_start_date.__str__().strip(ZERO_TIME)))
        # revise the end date of all of the data
        if len(end_date_changes) != 0:
            self.train_end_date = min(end_date_changes)
            for company in self.companies:
                company.revise_end_date(self.train_end_date)
            print("Data did not exist for every ticker at end date of {}; revising to the earliest ending time "
                  "(common among all companies' data) of {}".format(train_end_date.__str__().strip(ZERO_TIME),
                                                                    self.train_end_date.__str__().strip(ZERO_TIME)))
        self.start_date_str = self.train_start_date.__str__().strip(ZERO_TIME)
        self.end_date_str = self.train_end_date.__str__().strip(ZERO_TIME)

        # sting that describes the parameters for this model such that files for weights can be successfully loaded
        if self.num_companies > 1:
            considering_string = "_CONSIDERING_" + "&".join(list(map(lambda company:
                                                                     company.ticker, self.companies[1:])))
        else:
            considering_string = ""
        self.identifier = "MODEL_FOR_" + self.companies[0].ticker + considering_string + \
                          "_WITH_lstm_hidden_size_{}_lstm_num_layers_{}_input_size_{}_sequence_" \
                          "segment_length_{}".format(
                              self.lstm_hidden_size,
                              self.lstm_num_layers,
                              self.num_companies,
                              self.sequence_segment_length)

        self.model_weights_path = os.path.join(os.getcwd(), ".cache", self.identifier + ".bin")

        # initialize objects used during forward pass
        self.lstm = nn.LSTM(input_size=self.num_companies, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, dropout=self.drop_prob, batch_first=True)
        self.post_lstm_dropout = nn.Dropout(p=self.drop_prob)
        self.fc_1 = nn.Linear(self.lstm_hidden_size, 10)
        self.fc_2 = nn.Linear(10, self.num_companies)
        self.tanh = nn.Tanh()
        # self.rescaler = Rescaler(-0.5, 0.5)

        # initialize attributes with placeholder arrays
        self.daily_stock_data = np.array(0)
        self.train_sample_indices = np.array(0)
        self.test_sample_indices = np.array(0)
        self.train_loader_len = 0
        self.test_loader_len = 0
        self.data_len = 0

        # initialize optimizer and loss
        self.loss = nn.MSELoss()

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
        Creates a simple line plot of the entire dataset

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
            daily_stock_data.append(company.return_numpy_array_of_company_daily_stock_percent_change())
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
        and :attr:`lstm_num_layers`; the output is then passed through 2 fully connected layers such that the final
        number of features is the same as the input number of features (:attr:`num_companies`)

        :param X: input matrix of data of shape: (batch size, features (number of companies), sequence length)
        :param predict_beyond: number of days to recursively predict beyond the given input sequence
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
        :attr:`test_loader` at the end of each epoch. The forward pass through the network produces sequences of the
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

    def make_prediction_with_validation(self, predict_beyond: int = 30, num_plots: int = 2,
                                        data_start_indices: np.ndarray = None):
        r"""
        Selects data from the dataset and makes a prediction ``predict_beyond`` days out, and the actual values
        of the stock are shown alongside.

        :param predict_beyond: days to predict ahead in the future
        :param data_start_indices: indices corresponding to locations in the total dataset sequence for the training
        data to be gathered from (with the training data being of length :attr:`sequence_segment_length`)
        :return: length of the data being returned (training + prediction sequences)
        :return: datetime objects corresponding to data_start_indices
        :return: datetime objects corresponding to the end of the returned sequences
        :return: indices corresponding to the days where the predicted sequence starts
        :return: input and label sequence data associated with each pass of the model
        :return: numpy array of the model output
        :return: training data (in absolute stock value form instead of the % change that the model sees)
        :return: output prediction of the model converted from % change to actual stock values
        :return: label data (in absolute stock value form instead of % change) to compare to the output prediction
        :return: disparity between predicted stock values and actual stock values
        """
        input_and_pred_len = self.sequence_segment_length + predict_beyond
        if data_start_indices is None:
            data_start_indices = np.random.choice(self.daily_stock_data.shape[1] - input_and_pred_len, num_plots)

        start_train_datetimes = []  # holds datetime objects corresponding to the data_start_indices
        end_pred_datetimes = []  # holds datetime objects corresponding to the data_start_indices
        pred_data_start_indicies = []  # indices for the start of predictions
        train_and_actual_data = torch.zeros((num_plots, self.num_companies, input_and_pred_len))
        for i in range(num_plots):
            train_and_actual_data[i, :, :] = torch.from_numpy(
                self.daily_stock_data[:, data_start_indices[i]:data_start_indices[i] + input_and_pred_len])
            start_train_datetimes.append(self.companies[0].get_date_at_index(data_start_indices[i]))
            end_pred_datetimes.append(self.companies[0].get_date_at_index(data_start_indices[i] + input_and_pred_len))
            pred_data_start_indicies.append(data_start_indices[i] + input_and_pred_len - predict_beyond)

        # pass in the data for training
        output_numpy = self.forward(train_and_actual_data[:, :, :-(predict_beyond + 1)], predict_beyond).detach().\
            numpy()

        orig_stock_list = []
        pred_stock_list = []
        actual_stock_list = []
        disparity_list = []
        for i in range(num_plots):
            orig_stock_list.append(self.companies[0].data_frame["Close"].iloc[
                                       list(range(data_start_indices[i], pred_data_start_indicies[i], 1))])
            pred_stock_list.append(
                self.companies[0].reconstruct_stock_from_percent_change(output_numpy[i, 0, -predict_beyond:],
                                                                        initial_condition_index=(
                                                                                pred_data_start_indicies[
                                                                                    i] - 1))[1:])
            actual_stock_list.append(self.companies[0].data_frame["Close"].iloc[
                list(range(pred_data_start_indicies[i], pred_data_start_indicies[i] + predict_beyond))].values)
            disparity_list.append(np.abs(pred_stock_list[i] - actual_stock_list[i]))

        return input_and_pred_len, start_train_datetimes, end_pred_datetimes, pred_data_start_indicies, \
               train_and_actual_data, output_numpy, orig_stock_list, pred_stock_list, actual_stock_list, disparity_list

    def check_sliding_window_valid_at_index(self, end_pred_index, pred_beyond_range):
        r"""
        Checks that the index parameter for creating a distribution of predictions is valid for the dataset, and
        modifies it if it isn't (as well as prints a warning describing the condition)

        :param end_pred_index: index of the date that is desired to be predicted
        :param pred_beyond_range: tuple containing the range of the number of forecasted days the model will use to
        arrive at a prediction at ``end_pred_index``
        :return: end_pred_index (modified if necessary)
        """
        if end_pred_index is None:
            print("latest_data_index is None, so will set to minimum possible value")
            end_pred_index = self.sequence_segment_length + pred_beyond_range[1]
        if end_pred_index - (self.sequence_segment_length + (pred_beyond_range[1] - pred_beyond_range[0])) < 0:
            print("WARNING: latest_data_index, when combined with the provided pred_beyond_range, will yield negative"
                  "indices for training data start points; revising to smallest possible value")
            end_pred_index = self.sequence_segment_length + pred_beyond_range[1]
        if end_pred_index >= self.data_len:
            print("WARNING: latest_data_index is too large for dataset; revising to largest possible value")
            end_pred_index = self.data_len - 1
        return end_pred_index

    def generate_predicted_distribution(self, end_pred_index: int = None, pred_beyond_range: (int, int) = (1, 10)):
        r"""
        Returns a list of predicted stock values at a given date using a range of forecast lengths

        :param end_pred_index: index of the date that is desired to be predicted
        :param pred_beyond_range: tuple containing the range of the number of forecasted days the model will use to
        arrive at a prediction at ``end_pred_index``
        :return: list of predicted values (of length given by ``pred_beyond_range``)
        :return: actual stock value corresponding to the predictions
        """
        end_pred_index = self.check_sliding_window_valid_at_index(end_pred_index, pred_beyond_range)
        pred_beyond_range_delta = pred_beyond_range[1] - pred_beyond_range[0]
        predicted_value_list = []
        debug = []
        for i in range(pred_beyond_range[0], pred_beyond_range[1]):
            # the start of the desired index is the end value index decreased by the length of the prediction and the
            # training sequence length; the start index is then shifted back as the number of days that is predicted
            # beyond increases
            _, _, _, _, _, _, _, pred_stock_list, actual_stock_list, _ = self.make_prediction_with_validation(i,
                            num_plots=1, data_start_indices=np.array([end_pred_index - self.sequence_segment_length -
                                                                      pred_beyond_range_delta - i]))
            predicted_value_list.append(pred_stock_list[0][-1])
            debug.append(actual_stock_list[0][-1])
        return predicted_value_list, actual_stock_list[0][-1]

    def pred_in_conj(self, start_of_pred_idx: int, n_days: int, pred_beyond_range: (int, int) = (1, 10)):
        r"""
        Calls :method:`generate_predicted_distribution` to create a list of predictions for each day given in a given
        range, and returns the mean and standard deviation associated with each day.

        :param start_of_pred_index: integer corresponding to the first date whose distribution will be predicted
        :param n_days: number of days from ``start_of_pred_index`` to predict out
        :param pred_beyond_range: tuple containing the range of the number of forecasted days the model will use to
        arrive at a prediction at ``end_pred_index``
        :return: list of length ``n_days`` of the mean values associated with each day's predicted stock
        :return: list of length ``n_days`` of the standard deviation associated with each day's predicted stock
        """
        mean_list = []
        std_list = []
        for n in range(n_days):
            end_pred_index = start_of_pred_idx + n
            end_pred_index = self.check_sliding_window_valid_at_index(end_pred_index, pred_beyond_range)
            predicted_value_list, actual_value = self.generate_predicted_distribution(end_pred_index,
                                                                                      pred_beyond_range)
            mean_list.append(np.mean(predicted_value_list))
            std_list.append(np.std(predicted_value_list))
        return mean_list, std_list

    def plot_predicted_distribution(self, latest_data_index: int = None, pred_beyond_range: (int, int) = (1, 10)):
        r"""
        TODO: documentation
        """
        predicted_value_list, actual_value = self.generate_predicted_distribution(latest_data_index, pred_beyond_range)
        n_bins = round((pred_beyond_range[1] - pred_beyond_range[0]) / 3)
        if n_bins < 3:
            n_bins = 3
        plt.hist(predicted_value_list, bins=n_bins, color="green")
        plt.plot([actual_value, actual_value], [0, pred_beyond_range[1] - pred_beyond_range[0]], "-")
        plt.show()

    def plot_prediction_with_validation(self, predict_beyond: int = 30, num_plots: int = 5, plt_scl=20):
        r"""
        A method for debugging/validating :attr:`make_prediction_with_validation` - makes predictions and shows the
        raw output of the model, reconstructed stock prices, and disparity between predicted stock prices and actual
        stock prices.

        :param predict_beyond: days to predict ahead in the future
        :param num_plots: number of times to call :attr:`make_prediction_with_validation` and plot the results
        :plt_scl: integer for width and heigh parameters of matplotlib plot
        """
        forward_seq_len, start_dates, end_dates, pred_data_start_indicies, make_pred_data, \
        output_numpy, orig_stock_list, pred_stock_list, actual_stock_list, disparity_list = \
            self.make_prediction_with_validation(predict_beyond, num_plots)
        pred_beyond_plot_indices = np.arange(self.sequence_segment_length, forward_seq_len)
        pred_over_input_plot_indices = np.arange(1, self.sequence_segment_length)
        input_plot_indices = np.arange(0, self.sequence_segment_length)
        disparity_plot_indices = np.arange(1, predict_beyond + 1)
        _, axes = plt.subplots(num_plots, 3, figsize=(plt_scl, plt_scl))
        for ax in range(num_plots):
            axes[ax][0].plot(pred_beyond_plot_indices, output_numpy[ax, 0, -predict_beyond:], color='green',
                             label="pred. beyond",
                             linestyle='--', marker='o')
            axes[ax][0].plot(pred_over_input_plot_indices, output_numpy[ax, 0, :self.sequence_segment_length - 1],
                             color='gray',
                             label="pred. over input", linestyle='--', marker='o')
            axes[ax][0].plot(input_plot_indices,
                             make_pred_data.detach().numpy()[ax, 0, :self.sequence_segment_length], color='red',
                             label="input", linestyle='-', marker='o')
            axes[ax][0].plot(pred_beyond_plot_indices,
                             make_pred_data.detach().numpy()[ax, 0, self.sequence_segment_length:], label="actual",
                             linestyle='-', marker='o')
            axes[ax][0].set_title("{} % change from\n{} to {}".format(self.companies[0].ticker,
                                                                      start_dates[ax], end_dates[ax]))
            axes[ax][0].set_xlabel("Business days since {}".format(start_dates[ax]))
            axes[ax][0].set_ylabel("% Change")
            axes[ax][0].legend()
            axes[ax][1].plot(pred_beyond_plot_indices, pred_stock_list[ax], color="green",
                             label="pred. beyond", linestyle='--', marker='o')
            axes[ax][1].plot(input_plot_indices, orig_stock_list[ax], color='red', label="input", linestyle='-', marker='o')
            axes[ax][1].plot(pred_beyond_plot_indices, actual_stock_list[ax], label="actual", linestyle='-', marker='o')
            axes[ax][1].set_title("{} stock from\n{} to {}".format(self.companies[0].ticker, start_dates[ax],
                                                                   end_dates[ax]))
            axes[ax][1].set_xlabel("Business days since {}".format(start_dates[ax]))
            axes[ax][1].set_ylabel("Stock Price")
            axes[ax][1].legend()
            axes[ax][2].plot(disparity_plot_indices, disparity_list[ax], label="disparity",
                             linestyle="", marker="o")
            axes[ax][2].set_title("Disparity of Predicted and Actual Stock")
            axes[ax][2].set_xlabel("Num. predicted days out {}".format(start_dates[ax]))
            axes[ax][2].set_ylabel("Absolute difference between\nprediction and reality")
        plt.show()


if __name__ == "__main__":
    model: StockRNN

    # set to switch between loading saved weights if available
    try_load_weights = True

    model = StockRNN("AAPL", to_compare=["GOOGL", "MSFT", "MSI"], train_start_date=datetime(2012, 1, 1),
                     train_end_date=datetime(2019, 1, 1), try_load_weights=try_load_weights)
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

    # model.eval()
    model.plot_prediction_with_validation()
    # model.plot_predicted_distribution(12)
    model.pred_in_conj(123, 4)
