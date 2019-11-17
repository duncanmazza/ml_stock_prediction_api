"""
Code to train the RNN
(note: adapted from https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/)
TODO: remove the above 'adapted from...' note when the code becomes sufficiently different

@author: Duncan Mazza
"""

import torch.nn as nn
import torch
import numpy as np
from datetime import datetime
from src.get_data.pandas_stock_data import numpy_array_of_company_daily_stock_close

DEVICE = "cuda"  # selects the gpu to be used


class StockRNN(nn.Module):
    r"""
    Class for handling all RNN operations.
    """
    daily_stock_data: object

    def __init__(self, lstm_input_size: int, lstm_output_size: int, lstm_num_layers: int, ticker: str,
                 start_date: datetime = datetime(2017, 1, 1), end_date: datetime = datetime(2018, 1, 1),
                 sequence_length: int = 20, drop_prob: float = 0.5, device: str = DEVICE):
        r"""

        :param output_size:
        :param embedding_dim:
        :param hidden_dim:
        :param n_layers:
        :param drop_prob:
        """
        super(StockRNN, self).__init__()

        # params
        self.lstm_input_size = lstm_input_size
        self.lstm_output_size = lstm_output_size
        self.lstm_num_layers = lstm_num_layers
        self.drop_prob = drop_prob
        self.device = device
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length

        # initialize objects used during forward pass
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_output_size, self.lstm_num_layers, dropout=self.drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # self.conv1 = nn.Conv1d()
        # self.sigmoid = nn.Sigmoid()

        self.daily_stock_data = np.array(0)

    def forward(self, x, hidden):
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
        self.daily_stock_data = numpy_array_of_company_daily_stock_close(self.ticker, self.start_date, self.end_date)
        mod = len(self.daily_stock_data) % self.sequence_length
        if mod != 0 and truncate:
            self.daily_stock_data = self.daily_stock_data[:-mod]



output_size = 1
embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = StockRNN(output_size, embedding_dim, hidden_dim, n_layers)
model.to(DEVICE)  # send to GPU

lr = 0.005
criterion = nn.BCELoss()  # binary cross-entropy loss between target and output
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 1000
clip = 5
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(DEVICE), lab.to(DEVICE)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).'
                      'Saving model ...'.format(valid_loss_min, np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
