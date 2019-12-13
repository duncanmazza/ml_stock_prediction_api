"""
Code for the combined model approach.

@author: Shashank Swaminathan
"""

from src.BayesReg import CashMoneySwag
from src.StockRNN import StockRNN
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date

ZERO_TIME = " 00:00:00"

DEVICE = "cuda"  # selects the gpu to be used
TO_GPU_FAIL_MSG = "Unable to successfully run model.to('{}'). If running in Collaboratory, make sure " \
                  "that you have enabled the GPU your settings".format(DEVICE)

class WomboCombo:
    r"""
    Class for handling combined model operations.
    """
    def __init__(self, ticker, comp_tickers):
        r"""
        init function. It will set up the StockRNN and CashMoneySwag classes.

        :param ticker: Ticker of stocks to predict
        :param comp_tickers: List of tickers to compare desired ticker against. Used for StockRNN only.
        """
        self.srnn = StockRNN(ticker, to_compare=comp_tickers,
                             train_start_date=datetime(2012, 1, 1),
                             train_end_date=datetime.today(),
                             try_load_weights=False)
        self.cms = CashMoneySwag(ticker)

    def train(self, start_date, pred_start, pred_end, mw=0.5, n_epochs=10):
        r"""
        Main training function. It runs both the LSTM and GP models and stores results in attributes.

        :param start_date: Training start date (for GP model only). Provide as datetime object.
        :param pred_start: Date to start predictions from. Provide as datetime object.
        :param pred_end: Date to end predictions. Provide as datetime object.
        :param mw: Model weight. Used to do weighted average between GP and LSTM. 0 is for only the LSTM, and 1 is for only the GP. Defaults to 0.5 (equal split).
        :param n_epochs: Number of epochs to train the LSTM. Defaults to 10.

        :returns: (Mean predictions [t, y], Upper/lower bounds of 2 std [t, y])
        """
        dt_ps = date(pred_start.year, pred_start.month, pred_start.day)
        dt_pe = date(pred_end.year, pred_end.month, pred_end.day)
        self.n_days_pred = np.busday_count(dt_ps, dt_pe) + 1

        self.train_end = pred_start - pd.Timedelta(1, "D")
        return self._combo_shot(start_date, pred_start, pred_end,
                                mw = mw, n_epochs = n_epochs)

    def _combo_shot(self, start_date, pred_start, pred_end, mw=0.5, n_epochs=10):
        r"""
        Helper function to actually do the combo model training. Runs the two models individually, aligns the two results in time, then adds the two generated distributions as a weighted sum. Sets attribute combo_vals equal to the result.

        :param start_date: Training start date (for GP model only). Provide as datetime object.
        :param pred_start: Date to start predictions from. Provide as datetime object.
        :param pred_end: Date to end predictions. Provide as datetime object.
        :param mw: Model weight. Used to do weighted average between GP and LSTM. 0 is for only the LSTM, and 1 is for only the GP. Defaults to 0.5 (equal split).
        :param n_epochs: Number of epochs to train the LSTM. Defaults to 10.
        """
        self._srnn_train(pred_start, self.n_days_pred, n_epochs = n_epochs)
        self._cms_train(start_date, self.train_end, pred_end)
        m_combo = self.m_cms[-self.n_days_pred:]*(mw)+self.m_srnn*(1-mw)
        std_combo = self.std_cms[-self.n_days_pred:]*(mw)+self.std_srnn*(1-mw)

        xy_pred = [self.times, m_combo]
        upper = m_combo + 2*std_combo
        lower = m_combo - 2*std_combo
        band_x = np.append(self.times, self.times[::-1])
        band_y = np.append(lower, upper[::-1])
        std_bounds = [band_x, band_y]
        self.combo_vals = (xy_pred, std_bounds)

    def _srnn_train(self, pred_start, n_days_pred, n_epochs=10):
        r"""
        Helper function to train the LSTM using the StockRNN class. Generates upper and lower bounds of prediction based on mean and std. deviation. Sets attribute srnn_vals equal to result. Result is of form: ([time, mean prediction], [time, upper/lower bounds], [time, actual data prior to prediction], [time, actual data during prediction]).

        :param pred_start: Date to start predictions from. Provide as datetime object.
        :param n_days_pred: Number of days to predict ahead. Will only predict on business days.
        :param n_epochs: Number of epochs to train the LSTM. Defaults to 10.
        """
        srdf = self.srnn.companies[0].data_frame
        srdfdt = pd.to_datetime(srdf.Date)
        raw_p_st_idx = srdfdt.searchsorted(pred_start)
        p_st_idx = raw_p_st_idx + srdf.index[0]
        raw_p_e_idx = raw_p_st_idx + self.n_days_pred
        try:
            self.srnn.to(DEVICE)
            self.srnn.__togpu__(True)
        except RuntimeError:
            print(TO_GPU_FAIL_MSG)
        except AssertionError:
            print(TO_GPU_FAIL_MSG)
            self.srnn.__togpu__(False)

        self.srnn.do_training(num_epochs=n_epochs)
        self.m_srnn, self.std_srnn = self.srnn.pred_in_conj(p_st_idx, n_days_pred)
        self.times = srdf.Date.iloc[raw_p_st_idx:raw_p_e_idx]
        self.m_srnn = np.array(self.m_srnn)
        self.std_srnn = np.array(self.std_srnn)

        times_td = srdf.Date.iloc[raw_p_st_idx-50:raw_p_st_idx-1]
        td_srnn = srdf.Close.iloc[raw_p_st_idx-50:raw_p_st_idx-1]
        a_srnn = srdf.Close.iloc[raw_p_st_idx:raw_p_e_idx]

        xy_pred = [self.times, self.m_srnn]
        upper = self.m_srnn + 2*self.std_srnn
        lower = self.m_srnn - 2*self.std_srnn
        band_x = np.append(self.times, self.times[::-1])
        band_y = np.append(lower, upper[::-1])
        std_bounds = [band_x, band_y]
        train_data = [times_td, td_srnn]
        test_data = [self.times, a_srnn]
        self.srnn_vals = (xy_pred, std_bounds, train_data, test_data)

    def _cms_train(self, start_date, train_end, pred_end):
        r"""
        Helper function to train the GP model using the CashMoneySwag class. Sets attribute cms_vals equal to result. Result is of form: ([time, mean prediction], [time, upper/lower bounds], [time, actual data prior to prediction], [time, actual data during prediction]).

        :param start_date: Training start date (for GP model only). Provide as datetime object.
        :param train_end: Date to end training. Provide as datetime object.
        :param pred_end: Date to end predictions. Provide as datetime object. Assumes predictions begin right after training.
        """
        xy_pred, std_bounds, train_data, test_data = self.cms.go(start_date=start_date,
                                                                 split_date=train_end,
                                                                 end_date=pred_end)
        self.m_cms = xy_pred[1]
        self.std_cms = xy_pred[2]
        self.cms_vals = (xy_pred, std_bounds, train_data, test_data)
