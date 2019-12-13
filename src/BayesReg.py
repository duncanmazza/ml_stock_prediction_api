"""
Code for the Gaussian Process Model implementation.

@author: Shashank Swaminathan
"""
from datetime import datetime
import pymc3 as pm
import pandas as pd
import numpy as np
import warnings
import theano.tensor as tt
from src.get_data import Company

class GPM():
    r"""
    Class encapsulating Gaussian Process model implementation. Depends on the PyMC3 library.
    """
    def __init__(self, ticker, start_date=datetime(2000,1,1), end_date=datetime.today()):
        r"""
        init function. Fetches and prepares data from desired ticker using Company class from accompanying get_data API. Assumes start and end dates of January 1st, 2000 -> today.

        :param ticker: Ticker of stock to predict.
        :param start_date: Datetime object of when to start retrieving stock data. Defaults to 1/1/2000.
        :param end_date: Datetime object of when to stop retrieving stock data. Defaults to today's date.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        self.comp = Company(ticker, self.start_date, self.end_date)
        self.data = self._prep_data()
        self.split_date = self.data_early = self.data_later = None

    def _prep_data(self):
        r"""
        Helper function to prepare the data. Assumes an attribute named comp exists, and is a Company class.

        :returns: Dataframe of pruned data. Of form [times, normalized stock vals, raw stock vals]
        """
        self.comp.data_frame.index = pd.to_datetime(self.comp.data_frame.Date)
        raw_y = self.comp.return_numpy_array_of_company_daily_stock_close()
        norm_y = (raw_y-raw_y[0])/np.std(raw_y)
        t = self._dates_to_idx(self.comp.data_frame.index)
        return pd.DataFrame(data={'t':t, 'norm_y':norm_y, 'raw_y': raw_y},index=self.comp.data_frame.index)

    def go(self, start_date=None,split_date=pd.to_datetime("2019-09-30"),end_date=None):
        r"""
        Main function to train the model and predict future data. First generates training and testing data, then trains the GP on the training data. Finally, it predicts on the time range specified.

        :param start_date: Date to start training GP. If left as None, defaults to starting date of training data set.
        :param split_date: Date to end training.
        :param start_date: Date to end predicting using GP. If left as None, defaults to ending date of training data set. Assumes prediction starts right after training.

        :returns: A tuple of four arrays, containing the mean predictions, upper/lower bounds of predictions, training data, and test data for comparison. Predictions done on non business days are dropped.
        """
        print("Generating data...")
        self._gen_test_train_data(start_date, split_date, end_date)
        print("Training GPM...")
        self._train_gp()
        self._predict_gp()
        print("Generating plot...")
        return self._get_plot_vals()

    def _gen_test_train_data(self, start_date, split_date, end_date):
        r"""
        Helper function to generate the training and testing data.

        :param start_date: Date to start training GP. If left as None, defaults to starting date of training data set.
        :param split_date: Date to end training.
        :param start_date: Date to end predicting using GP. If left as None, defaults to ending date of training data set. Assumes prediction starts right after training.
        """
        if start_date==None:
            start_date=self.start_date
        if end_date==None:
            end_date=self.end_date
        self.train_start = start_date
        self.test_end = end_date
        self.split_date=split_date
        start_idx = self.data.index.searchsorted(start_date)
        sep_idx = self.data.index.searchsorted(split_date)+1 # Y-M-D
        end_idx = self.data.index.searchsorted(end_date)
        self.data_early = self.data.iloc[start_idx:sep_idx, :]
        self.data_later = self.data.iloc[sep_idx:end_idx+1, :]

    def _train_gp(self):
        r"""
        Helper function to train the GP model. Uses a combo of 3 GPs, with ExpQuad, Rational, and Periodic kernels. Noise is modeled as a combo of WhiteNoise and a Matern32 kernel. MAP is stored in attribute named mp. Conditioned GP is stored in attribute named gp.
        """
        with pm.Model() as model:
            # yearly periodic component x long term trend
            η_per = pm.HalfCauchy("η_per", beta=0.75, testval=1.0)
            period  = pm.Normal("period", mu=1, sigma=0.05)
            ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=4, beta=3)
            cov_seasonal = η_per**2 * pm.gp.cov.Periodic(1, period, ℓ_psmooth)
            gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

            # small/medium term irregularities
            η_med = pm.HalfCauchy("η_med", beta=0.5, testval=0.1)
            ℓ_med = pm.Gamma("ℓ_med", alpha=4, beta=7)
            α = pm.Gamma("α", alpha=5, beta=2)
            cov_medium = η_med**2 * pm.gp.cov.RatQuad(1, ℓ_med, α)
            gp_medium = pm.gp.Marginal(cov_func=cov_medium)

            # long term trend
            η_trend = pm.HalfCauchy("η_trend", beta=1, testval=2.0)
            ℓ_trend = pm.Gamma("ℓ_trend", alpha=7, beta=3)
            cov_trend = η_trend**2 * pm.gp.cov.ExpQuad(1, ℓ_trend)
            gp_trend = pm.gp.Marginal(cov_func=cov_trend)

            # noise model
            η_noise = pm.HalfNormal("η_noise", sigma=0.5, testval=0.05)
            ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
            σ  = pm.HalfNormal("σ",  sigma=0.25, testval=0.05)
            cov_noise = η_noise**2 * pm.gp.cov.Matern32(1, ℓ_noise) +\
                        pm.gp.cov.WhiteNoise(σ)

            # The Gaussian process is a sum of these three components
            self.gp = gp_seasonal + gp_medium + gp_trend

            # Since the normal noise model and the GP are conjugates
            # We use `Marginal` with the `.marginal_likelihood` method
            t=self.data_early['t'].values[:,None]
            y=self.data_early['norm_y'].values
            y_ = self.gp.marginal_likelihood("y", X=t, y=y, noise=cov_noise)

            # this line calls an optimizer to find the MAP
            self.mp = pm.find_MAP(include_transformed=True)

    def _predict_gp(self):
        r"""
        Helper function for predicting over specified range. Converts back from normalized values to actual values. Stores result of predictions in attribute named fit.
        """
        # predict at a 1 day granularity using generated datetime array.
        dates = pd.date_range(start=self.train_start, end=self.test_end, freq="1D")
        tnew = self._dates_to_idx(dates)[:,None]
        std_y=np.std(self.data['raw_y'].values)
        first_y=self.data['raw_y'][0]

        print("Predicting with GPM ...")
        mu, var = self.gp.predict(tnew, point=self.mp, diag=True)

        # Convert back from normalized values to raw values.
        mean_pred = mu*std_y + first_y
        var_pred  = var*std_y**2

        # make dataframe to store fit results
        self.fit = pd.DataFrame({"t": tnew.flatten(),
                                 "mu_total": mean_pred,
                                 "sd_total": np.sqrt(var_pred)},
                                index=dates)

    def _get_plot_vals(self):
        r"""
        Helper function to get useful values for plotting. Predictions done on non business days are dropped.

        :returns: A tuple of four arrays, containing the mean predictions, upper/lower bounds of predictions, training data, and test data for comparison.
        """
        fit = self.fit

        # Search for and remove predictions on non-business days from fit
        k=0
        l=[]
        for i in fit.index.values:
            ts = (i - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            j=datetime.utcfromtimestamp(ts)
            if (j.weekday() == 5 or j.weekday() == 6):
                l.append(k)
            k = k + 1
        fit.drop(fit.index[l])

        # Generate upper and lower bound predictions, 2 std deviations away from mean.
        upper = fit.mu_total.values + 2*fit.sd_total.values
        lower = fit.mu_total.values - 2*fit.sd_total.values
        band_x = np.append(fit.index.values, fit.index.values[::-1])
        band_y = np.append(lower, upper[::-1])

        # Generate output data.
        xy_pred = [fit.index.values, fit.mu_total.values, fit.sd_total.values]
        std_bounds = [band_x, band_y]
        train_data = [self.data_early.index, self.data_early['raw_y']]
        test_data = [self.data_later.index, self.data_later['raw_y']]
        return xy_pred, std_bounds, train_data, test_data

    def _plot_stock_data(self,show_split=False,raw_data=True,rg=None):
        r"""
        Helper function that plots stock data for desired ticker using Bokeh. Only useful in Jupyter Notebooks.

        :param show_split: If true, checks if training and testing data split has been generated. If so, will plot a yellowed region indicating split. If not, will generate warning. If false, will not do anything related.
        :param raw_data: If true, will plot raw data. If false, will plot normalized data.
        :param rg: Array of [start date, end date]. If left None, will default to starting date of stock value data set.
        """
        # If no range specified, assume whole thing
        if rg==None:
            rg=[self.start_date,self.end_date]
        # Prep data to plot
        s_idx=self.data.index.searchsorted(rg[0])
        e_idx=self.data.index.searchsorted(rg[1])
        sel_dat=self.data.iloc[s_idx:e_idx,:]
        t=sel_dat.index
        if raw_data:
            y=sel_dat['raw_y']
        else:
            y=sel_dat['norm_y']
        # Plot data
        p = figure(x_axis_type='datetime', title='Prices of '+self.ticker+' stock over time',
                   x_range=rg, plot_width=900, plot_height=573)
        p.yaxis.axis_label = 'Stock price'
        p.xaxis.axis_label = 'Date'
        p.line(t,y,line_width=2, line_color="black", alpha=0.5)
        # Add box region to show test/train split
        if show_split and self.split_date==None:
            warnings.warn('Test/train split on data has not yet been performed.'+\
                          ' Ignoring request for test region highlighting.')
        elif show_split and self.split_date!=None:
            predict_region = BoxAnnotation(left=self.split_date,
                               fill_alpha=0.1, fill_color="yellow")
            p.add_layout(predict_region)
        show(p)

    def _dates_to_idx(self, tlist):
        r"""
        Helper function to convert datetime objects into integers useful for training.

        :param tlist: Numpy array or Pandas Series of datetime objects.

        :returns: Numpy array of integers representing datetimes.
        """
        reference_time = self.start_date
        t = (tlist - reference_time) / pd.Timedelta(30, "D")
        return np.asarray(t)


# if __name__ == "__main__":
#     cms = GPM('AAPL')
#     rg=[datetime(2019,7,31),datetime(2019,9,30),datetime(2019,10,31)]
#     cms.gen_test_train_data(start_date=rg[0],split_date=rg[1],end_date=rg[2])
#     cms.train_gp()
#     cms.predict_gp()
#     xy_pred, std_bounds, train_data, test_data = cms.get_plot_vals()
