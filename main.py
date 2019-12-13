'''
Use the ``bokeh serve`` command to run the code by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
import pandas as pd
from datetime import datetime
from src.BayesReg import CashMoneySwag
from src.StockRNN import StockRNN
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure

DEVICE = "cuda"
TO_GPU_FAIL_MSG = "Unable to successfully run model.to('{}'). If running in Collaboratory, make sure " \
                  "that you have enabled the GPU your settings".format(DEVICE)


if __name__ == "__main__":
    # Define the company of interest whose stock price will bew predicted
    coi = "IBM"
    # Define the companies that will be included as features in the lstm dataset
    to_compare = ["HPE", "XRX", "ACN", "ORCL"]
    # Define the start and end dates for the whole data
    train_start_date = datetime(2012, 1, 1)
    train_end_date = datetime(2019, 1, 1)
    pred_start = datetime(2012, 1, 1)

    # Step 1. Instantiate the lstm and update the start and end dates
    lstm: StockRNN
    lstm = StockRNN(coi, to_compare=to_compare, train_start_date=train_start_date,
                    train_end_date=train_end_date, try_load_weights=True)
    train_start_date = lstm.train_start_date
    train_end_date = lstm.train_end_date
    try:
        lstm.to(DEVICE)
        lstm.__togpu__(True)
    except RuntimeError:
        print(TO_GPU_FAIL_MSG)
    except AssertionError:
        print(TO_GPU_FAIL_MSG)
        lstm.__togpu__(False)

    # Step 2. Train the lstm
    lstm.do_training(num_epochs=100)

    # Step 3. Instantiate the GPM
    cms = CashMoneySwag('AAPL')
    rg=[train_start_date, train_end_date, datetime(2019,10,31)]
    cms.gen_test_train_data(start_date=rg[0],split_date=rg[1],end_date=rg[2])
    cms.train_gp()
    cms.predict_gp()
    xy_pred, std_bounds, train_data, test_data = cms.get_plot_vals()

    # Set up plot
    p = figure(plot_height=572, plot_width=900,
                  title="my sine wave",x_axis_type='datetime',
                  tools="crosshair,pan,reset,save,wheel_zoom")
    p.yaxis.axis_label = 'money'
    p.xaxis.axis_label = 'time'

    # Set up source data
    sfit = ColumnDataSource(data=dict(x=xy_pred[0], y=xy_pred[1]))
    bfit = ColumnDataSource(data=dict(x=std_bounds[0], y=std_bounds[1]))
    defit = ColumnDataSource(data=dict(x=train_data[0], y=train_data[1]))
    dlfit = ColumnDataSource(data=dict(x=test_data[0], y=test_data[1]))

    # Plot total fit
    p.line('x','y',source=sfit,
           line_width=1, line_color="firebrick", legend="Total fit")
    p.patch('x', 'y',source=bfit,
            color="firebrick", alpha=0.6, line_color="white")

    # Plot true value
    p.circle('x','y',source=defit,
             color="black", legend="Training data")
    p.circle('x','y',source=dlfit,
             color="yellow", legend="Test data")
    p.legend.location = "top_left"


    # Set up widgets
    text = TextInput(title="title", value='my time stretcher')
    start_date = TextInput(title="Start Date", value='2018-10-1')
    num_month_train = Slider(title="Number of months to train on", value=1, start=1, end=10, step=0.5)
    num_month_pred = Slider(title="Number of months to predict", value=1, start=1, end=5, step=0.5)


    # Set up callbacks
    def update_title(attrname, old, new):
        p.title.text = text.value

    text.on_change('value', update_title)

    def update_data(attrname, old, new):

        # Get the current slider values
        t_start = pd.to_datetime(start_date.value)
        t_train = t_start + num_month_train.value*pd.Timedelta(15,"D")
        t_pred = t_train + num_month_pred.value*pd.Timedelta(15,"D")

    for w in [text, start_date, num_month_train, num_month_pred]:
        w.on_change('value', update_data)


    # Set up layouts and add to document
    inputs = column(text, start_date, num_month_train, num_month_pred)

    curdoc().add_root(row(inputs, p, width=800))
    curdoc().title = "Sliders"

