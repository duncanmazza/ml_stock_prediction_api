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
    def __init__(self, ticker, comp_tickers):
        self.srnn = StockRNN(ticker, to_compare=comp_tickers,
                             train_start_date=datetime(2012, 1, 1),
                             train_end_date=datetime.today(),
                             try_load_weights=True)
        self.cms = CashMoneySwag(ticker)

    def train(self, start_date, pred_start, pred_end, model_name="Combined", mw=0.5, n_epochs=10):
        dt_ps = date(pred_start.year, pred_start.month, pred_start.day)
        dt_pe = date(pred_end.year, pred_end.month, pred_end.day)
        self.n_days_pred = np.busday_count(dt_ps, dt_pe) + 1

        self.train_end = pred_start - pd.Timedelta(1, "D")
        return self._combo_shot(start_date, pred_start, pred_end,
                                mw = mw, n_epochs = n_epochs)

    def _combo_shot(self, start_date, pred_start, pred_end, mw=0.5, n_epochs=10):
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
        xy_pred, std_bounds, train_data, test_data = self.cms.go(start_date=start_date,
                                                                 split_date=train_end,
                                                                 end_date=pred_end)
        self.m_cms = xy_pred[1]
        self.std_cms = xy_pred[2]
        self.cms_vals = (xy_pred, std_bounds, train_data, test_data)
