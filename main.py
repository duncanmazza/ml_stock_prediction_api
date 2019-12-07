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

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Select
from bokeh.plotting import figure


# Select model type
m_type = 'GPM'

# Prep data
cms = CashMoneySwag('AAPL')
rg=[datetime(2019,7,31),datetime(2019,9,30),datetime(2019,10,31)]
xy_pred, std_bounds, train_data, test_data = cms.go(start_date=rg[0],split_date=rg[1],end_date=rg[2])

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
start_date = TextInput(title="Start Date", value='2019-07-31')
num_month_train = TextInput(title="Number of months to train on", value="2")
num_month_pred = TextInput(title="Number of months to predict", value="1")
model_type = Select(title="Model Type", options=["Combined","GPM","LSTM"], value="GPM")


# Set up callbacks
def update_title(attrname, old, new):
    p.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    t_start = pd.to_datetime(start_date.value)
    t_train = t_start + int(num_month_train.value)*pd.Timedelta(15,"D")
    t_pred = t_train + int(num_month_pred.value)*pd.Timedelta(15,"D")

    m_type = model_type.value

    xy_pred, std_bounds, train_data, test_data = cms.go(t_start,t_train,t_pred)
    sfit.data=dict(x=xy_pred[0], y=xy_pred[1])
    bfit.data=dict(x=std_bounds[0], y=std_bounds[1])
    defit.data=dict(x=train_data[0], y=train_data[1])
    dlfit.data=dict(x=test_data[0], y=test_data[1])

for w in [text, start_date, num_month_train, num_month_pred, model_type]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(text, model_type, start_date, num_month_train, num_month_pred)

curdoc().add_root(row(inputs, p, width=800))
curdoc().title = "Stock Price Predictions using the ML Stock Prediction API"

# if __name__ == "__main__":
#     x_train, y_train, x_pred, y_pred = func_of_dates(datetime(2017,10,1),datetime(2018,10,1),datetime(2019,10,1))
#     print(x,y_train, y_pred)
