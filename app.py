from flask import Flask, render_template
#from lstm import fetchAllData
from model import forcaster
import quandl
import numpy as np
import pandas as pd
from datetime import datetime
# , url_for


def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')

def fetchSpecificData(*, startDate,endDate,queryParams, **kwargs):
    if parser(startDate):
        pass
    else:
        return "invalid start date"
            
    if parser(endDate):
        pass
    else:
        return "invalid end date"
            
    allDataColumns=["BCHAIN/TOTBC","BCHAIN/MKTCP","BCHAIN/TRFEE","BCHAIN/TRFUS","BCHAIN/NETDF","BCHAIN/NTRAN","BCHAIN/NTRAT","BCHAIN/NTREP","BCHAIN/NADDU","BCHAIN/NTRBL","BCHAIN/TOUTV","BCHAIN/ETRAV","BCHAIN/ETRVU","BCHAIN/TRVOU","BCHAIN/TVTVR","BCHAIN/MKPRU","BCHAIN/CPTRV","BCHAIN/CPTRA","BCHAIN/HRATE","BCHAIN/MIREV","BCHAIN/ATRCT","BCHAIN/BCDDC","BCHAIN/BCDDE","BCHAIN/BCDDW","BCHAIN/BCDDM","BCHAIN/BCDDY","BCHAIN/BLCHS","BCHAIN/AVBLS","BCHAIN/MWTRV","BCHAIN/MWNUS","BCHAIN/MWNTD","BCHAIN/MIOPM","BCHAIN/DIFF"]
          
    result =  all(elem in allDataColumns   for elem in queryParams  )
 
    if result:
        pass   
    else :
        return "invalid Query Parameters"
 
    todaysDate=datetime.today().strftime('%Y-%m-%d')
    quandl.ApiConfig.api_key = 'idjLfySzftDM7Cn1oSGi'
    data=quandl.get(queryParams,start_date =startDate, end_date = endDate)
    data.rename(columns=lambda x: x[0:12], inplace=True)
          
    data["Date"]=data.index
 
    data=data[data["Date"].notnull()]
    data.fillna(method='ffill', inplace=True)
    data=data.dropna(thresh=len(data) - 3, axis=1)
 
    return data

todaysDate=datetime.today().strftime('%Y-%m-%d')
data=fetchSpecificData(startDate="2015-01-01", endDate=todaysDate,queryParams=["BCHAIN/MKPRU"])
data = data.dropna()
Open = data[["BCHAIN/MKPRU"]]

app = Flask(__name__)


@app.route('/')
def index():
    res = forecaster(Open, 30)
    

    return render_template('index.html', res=res)


@app.route('/wallet/')
def wallet():
    return render_template('wallet.html')


if __name__ == '__main__':
    app.run(debug=True)
