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
from src.CombinedModel import WomboCombo

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Select, BoxAnnotation
from bokeh.plotting import figure

# Select model type
m_type = 'GPM'

# Prep data
wc = WomboCombo(ticker='AAPL', comp_tickers=["GOOGL", "MSFT", "MSI"])
rg=[datetime(2019,5,31),datetime(2019,10,1),datetime(2019,10,31)]
wc.train(rg[0], rg[1], rg[2], mw=0.5, n_epochs=51)

# # Prep data
# cms = CashMoneySwag('AAPL')
# xy_pred, std_bounds, train_data, test_data = cms.go(start_date=rg[0],split_date=rg[1],end_date=rg[2])

# Set up plot
p = figure(plot_height=572, plot_width=900,
              title="Stock Price Prediction for AAPL",x_axis_type='datetime',
              tools="crosshair,box_zoom,pan,reset,save,wheel_zoom")
p.yaxis.axis_label = 'money'
p.xaxis.axis_label = 'time'

# Set up source data
wc_mfit = ColumnDataSource(data=dict(x=wc.combo_vals[0][0],y=wc.combo_vals[0][1]))
wc_sdfit = ColumnDataSource(data=dict(x=wc.combo_vals[1][0],y=wc.combo_vals[1][1]))
nn_mfit = ColumnDataSource(data=dict(x=wc.srnn_vals[0][0],y=wc.srnn_vals[0][1]))
nn_sdfit = ColumnDataSource(data=dict(x=wc.srnn_vals[1][0],y=wc.srnn_vals[1][1]))
gp_mfit = ColumnDataSource(data=dict(x=wc.cms_vals[0][0],y=wc.cms_vals[0][1]))
gp_sdfit = ColumnDataSource(data=dict(x=wc.cms_vals[1][0],y=wc.cms_vals[1][1]))
gp_tdfit = ColumnDataSource(data=dict(x=wc.cms_vals[2][0],y=wc.cms_vals[2][1]))
gp_adfit = ColumnDataSource(data=dict(x=wc.cms_vals[3][0],y=wc.cms_vals[3][1]))

# Plot total fit
wc_m=p.line('x','y',source=wc_mfit,
            line_width=1, line_color="green",
            legend_label="Combined Model Fit")
wc_sd=p.patch('x', 'y',source=wc_sdfit,
              color="lawngreen", alpha=0.6, line_color="white")

nn_m=p.line('x','y',source=nn_mfit,
            line_width=1, line_color="darkslateblue", legend_label="LSTM Model Fit")
nn_sd=p.patch('x', 'y',source=nn_sdfit,
              color="dodgerblue", alpha=0.6, line_color="white")

gp_m=p.line('x','y',source=gp_mfit,
            line_width=1, line_color="firebrick", legend_label="GP Model Fit")
gp_sd=p.patch('x', 'y',source=gp_sdfit,
              color="firebrick", alpha=0.6, line_color="white")

# Plot true value
td=p.circle('x','y',source=gp_tdfit,
            color="black", legend_label="Training data")
ad=p.circle('x','y',source=gp_adfit,
            color="yellow", legend_label="Actual data")
# predict_region = BoxAnnotation(left=rg[1],
#                                fill_alpha=0.1, fill_color="yellow")
# p.add_layout(predict_region)
p.legend.location = "top_left"


# Set up widgets
text = TextInput(title="title", value='Stock Price Prediction for AAPL')
start_date = TextInput(title="Start Date", value='2019-07-31')
num_month_train = TextInput(title="Number of months to train on", value="2")
num_month_pred = TextInput(title="Number of months to predict", value="1")
model_type = Select(title="Model Type",
                    options=["Combined","GPM","LSTM"], value="Combined")
m_weight = Slider(title="Percent weight of each model: 1 for GPM, 0 for LSTM",
                  value=0.5, start=0.1, end=1.0, step=0.1)
num_epochs = Slider(title="Number of training epochs for LSTM",
                    value=51, start=1, end=101, step=10)
# Set up callbacks
def update_title(attrname, old, new):
    p.title.text = text.value
text.on_change('value', update_title)

def update_model_type(attrname, old, new):
    m_type = model_type.value
    if (m_type == "GPM"):
        wc_m.visible = False
        wc_sd.visible = False
        nn_m.visible = False
        nn_sd.visible = False
        gp_m.visible = True
        gp_sd.visible = True
    elif (m_type == "LSTM"):
        wc_m.visible = False
        wc_sd.visible = False
        nn_m.visible = True
        nn_sd.visible = True
        gp_m.visible = False
        gp_sd.visible = False
    elif (m_type == "Combined"):
        wc_m.visible = True
        wc_sd.visible = True
        nn_m.visible = True
        nn_sd.visible = True
        gp_m.visible = True
        gp_sd.visible = True
model_type.on_change('value', update_model_type)

def update_data(attrname, old, new):
    # Get the current slider values
    t_start = pd.to_datetime(start_date.value)
    t_train = t_start + float(num_month_train.value)*pd.Timedelta(30,"D")
    pred_start = t_train + pd.Timedelta(1, "D")
    pred_end = pred_start + float(num_month_pred.value)*pd.Timedelta(30,"D")
    ne = num_epochs.value
    mw = m_weight.value

    # xy_pred, std_bounds, train_data, test_data = cms.go(t_start,t_train,t_pred)
    wc.train(t_start, pred_start, pred_end, mw=mw, n_epochs=ne)
    wc_mfit = ColumnDataSource(data=dict(x=wc.combo_vals[0][0],y=wc.combo_vals[0][1]))
    wc_sdfit = ColumnDataSource(data=dict(x=wc.combo_vals[1][0],y=wc.combo_vals[1][1]))
    nn_mfit = ColumnDataSource(data=dict(x=wc.srnn_vals[0][0],y=wc.srnn_vals[0][1]))
    nn_sdfit = ColumnDataSource(data=dict(x=wc.srnn_vals[1][0],y=wc.srnn_vals[1][1]))
    gp_mfit = ColumnDataSource(data=dict(x=wc.cms_vals[0][0],y=wc.cms_vals[0][1]))
    gp_sdfit = ColumnDataSource(data=dict(x=wc.cms_vals[1][0],y=wc.cms_vals[1][1]))
    gp_tdfit = ColumnDataSource(data=dict(x=wc.cms_vals[2][0],y=wc.cms_vals[2][1]))
    gp_adfit = ColumnDataSource(data=dict(x=wc.cms_vals[3][0],y=wc.cms_vals[3][1]))

for w in [text, start_date, num_month_train, num_month_pred, m_weight, num_epochs]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(text,
                model_type,
                start_date,
                num_month_train,
                num_month_pred,
                m_weight, num_epochs)

curdoc().add_root(row(inputs, p, width=800))
curdoc().title = "Stock Price Predictions using the ML Stock Prediction API"

# if __name__ == "__main__":
#     x_train, y_train, x_pred, y_pred = func_of_dates(datetime(2017,10,1),datetime(2018,10,1),datetime(2019,10,1))
#     print(x,y_train, y_pred)
