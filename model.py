import warnings
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.layers import LSTM, Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

warnings.filterwarnings("ignore")
from datetime import datetime

import chart_studio.plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as pyoff



def forecaster(Data, days_to_predict):
    Open = Data
    scaler = MinMaxScaler()
    scaler.fit(Open)
    train = scaler.transform(Open)

    n_input = days_to_predict
    n_features = 1
    
    #generate time series sequences for the forecast 
    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=8)

    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer="nadam", loss="mse",metrics=['accuracy'])
    history = model.fit_generator(generator,epochs=10)

    #evaluate the model
    score = model.evaluate(generator)

    # plt.figure(figsize=(15,5))
    # plt.plot(history.history['loss'])
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()

    #predict the data
    pred_train=model.predict(generator)
    pred_train=scaler.inverse_transform(pred_train)
    pred_train=pred_train.reshape(-1)

    pred_list = []
    
    batch = train[-n_input:].reshape((1, n_input, n_features))
    
    for i in range(n_input):   
        pred_list.append(model.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

    add_dates = [Open.index[-1] + DateOffset(days=x) for x in range(0, days_to_predict + 1) ]
    future_dates = pd.DataFrame(index=add_dates[1:],columns=Open.columns)

    #calculate the forecast
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                            index=future_dates[-n_input:].index, columns=['Forecast'])
    
    df_proj = pd.concat([Open,df_predict], axis=1)


    new_data = pd.DataFrame(df_proj["Forecast"])

    new_data['Date'] = new_data.index

    new_data.tail(10)
    forecast = [x for x in new_data["Forecast"]][-(days_to_predict):]
    date = [x for x in new_data["Date"].dt.strftime('%Y-%m-%d')][-(days_to_predict):]
    
    res = {} 
    for key in date: 
        for value in forecast: 
            res[key] = value
            forecast.remove(value) 
            break



    # plot_data = [
    #     go.Scatter(
    #         x=df_proj.index,
    #         y=df_proj['BCHAIN/MKPRU'],
    #         name='Actual'
    #     ),
    #     go.Scatter(
    #         x=df_proj.index,
    #         y=df_proj['Forecast'],
    #         name='Forecast'
    #     ),
    #     go.Scatter(
    #         x=df_proj.index,
    #         y=pred_train,
    #         name='Prediction'
    #     )
    # ]
    # plot_layout = go.Layout(
    #         title='Bitcoin stock price prediction'
    #     )
    # fig = go.Figure(data=plot_data, layout=plot_layout)
    # pyoff.iplot(fig)

    return res