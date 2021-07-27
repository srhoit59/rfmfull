import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import warnings
import statsmodels.api as sm
from fbprophet import Prophet
from pystan import *
import os
from io import StringIO,BytesIO
from azure.storage.blob import BlockBlobService
##reading input data from blob
blobservice = BlockBlobService(account_name='flaskstorage', account_key='M9Hax/c6wKCdVXIcmBafad35/ctWW2OQJQynRMrM29D+mfZXWW53MF0Sthsf0cmWN+/XukVg/aZQ/6XBAB4cgg==') 
byte_stream = BytesIO()
blobservice.get_blob_to_stream(container_name='htflaskcontainer', blob_name='mtsdu.xlsx', stream=byte_stream)
byte_stream.seek(0)
ser=pd.read_excel(byte_stream,index_col=0)
byte_stream.close()
ser.head()
#ser = pd.read_excel('Copy of DB-O.xlsx',sheet_name='MTSDB1-O',index_col=0)
#for i in range(0,4):
#    ser.iloc[:21,i]=ser.iloc[:21,i].apply(lambda x : x*1000)
#    print(i)
########################FBPROPHET####################
revdf = ser
revdf['ds']= revdf.index
revdf=revdf.rename(columns={"Total Sum of Revenue":'y'})
my_model = Prophet(interval_width=0.95, yearly_seasonality=True,changepoint_prior_scale=4)
my_model.fit(revdf[['ds','y']])
future_dates = my_model.make_future_dataframe(periods=6, freq='MS')
forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']]
from sklearn.metrics import mean_squared_error
rms = np.sqrt(mean_squared_error(revdf['y'],forecast['yhat'][:len(revdf['y'])]))
#print(rms)
adrf = forecast['yhat'].tail(6)
my_model.plot(forecast,uncertainty=True)
########################### predicting FTE ##################################
A=ser['Total Sum of BFTE']
#from plotly.plotly import plot_mpl
"""
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(A, model='additive',freq=12)
fig = result.plot()
"""
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    plt.figure(figsize=(12,8))
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    """
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    """
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(A)
 
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

grab = []
arm = []
warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(A,trend='n',exog=ser['Total Sum of Revenue'].values,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=True,
                                            enforce_invertibility=False)

            results = mod.fit()
            
            #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            grab.append(results.aic)
            arm.append([results.aic,param, param_seasonal])
        except:
            continue
arm.sort()

mod = sm.tsa.statespace.SARIMAX(A,trend='t',exog=ser['Total Sum of Revenue'].values,
                                order=arm[0][1],
                                seasonal_order=arm[0][2],
                                enforce_stationarity=True,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary())
"""
results.plot_diagnostics(figsize=(16, 8))
plt.show()
"""
pred = results.get_prediction(start=1, dynamic=False)
pred_ci = pred.conf_int()

y_forecasted = pred.predicted_mean
y_truth = A['2015-07-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

adft =pd.concat([y_forecasted,y_truth,ser['Total Sum of Revenue']],axis=1)
adft=adft.reset_index()
adft=adft.rename(columns={0:'PredictedBFTE'})

lst=[1,2,3,4,5,10,15,20,25]
date=forecast['ds'].tail(6).values

answersreg =[]
for i in range(len(lst)):
    for j in range(len(adrf)):
        #print(j)
        mul = (lst[i]/100)*adrf.iloc[j]
        #print(mul)
        final = adrf.iloc[j]+mul
        #print(final)
        c=np.array(final)
        c=c.reshape(1,1)
        predi = results.get_forecast(steps=1,exog=c)
        answersreg.append([date[j],final,predi.predicted_mean.tolist()[0]])
        
answersreg1 =[]
for i in range(len(lst)):
    for j in range(len(adrf)):
        #print(j)
        mul = (lst[i]/100)*adrf.iloc[j]
        #print(mul)
        final = adrf.iloc[j]-mul
        #print(final)
        c=np.array(final)
        c=c.reshape(1,1)
        predi = results.get_forecast(steps=1,exog=c)
        answersreg1.append([date[j],final,predi.predicted_mean.tolist()[0]])

answersreg2 =[]
for j in range(len(adrf)):
    #print(j)
    final = adrf.iloc[j]
    c=np.array(final)
    c=c.reshape(1,1)
    predi = results.get_forecast(steps=1,exog=c)
    answersreg2.append([date[j],final,predi.predicted_mean.tolist()[0]])
    
for i in range(len(lst)):
    #print(i)
    if i == 0:
        df1 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=0
        while h<=5:
            df1.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df1['Value']='INC1'
        df1 = df1.reset_index()
    elif i == 1:
        df2 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=6
        while h<=11:
            df2.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df2['Value']='INC2'
        df2 = df2.reset_index()    
    elif i == 2:
        df3 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=12
        while h<=17:
            df3.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df3['Value']='INC3'
        df3 = df3.reset_index()
    elif i == 3:
        df4 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=18
        while h<=23:
            df4.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df4['Value']='INC4'
        df4 = df4.reset_index()
    elif i == 4:
        df5 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=24
        while h<=29:
            df5.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df5['Value']='INC5'
        df5 = df5.reset_index()
    elif i == 5:
        df10 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=30
        while h<=35:
            df10.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df10['Value']='INC10'
        df10 = df10.reset_index()
    elif i == 6:
        df15 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=36
        while h<=41:
            df15.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df15['Value']='INC15'
        df15 = df15.reset_index()
    elif i == 7:
        df20 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=42
        while h<=48:
            df20.loc[answersreg[h][0]]=answersreg[h][1:]
            h=h+1
        df20['Value']='INC20' 
        df20 = df20.reset_index()
    elif i==8:
        df = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=0
        while h<=5:
            df.loc[answersreg2[h][0]]=answersreg2[h][1:]
            h=h+1
        df['Value']='Forecasted' 
        df = df.reset_index()   
        
    else:
        print('It didnt work')

ASGpos = pd.concat([df,df1,df2,df3,df4,df5,df10,df15,df20])

for i in range(len(lst)):
    #print(i)
    if i == 0:
        df1 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=0
        while h<=5:
            df1.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df1['Value']='DECR1'
        df1 = df1.reset_index()
    elif i == 1:
        df2 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=6
        while h<=11:
            df2.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df2['Value']='DECR2'
        df2 = df2.reset_index()    
    elif i == 2:
        df3 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=12
        while h<=17:
            df3.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df3['Value']='DECR3'
        df3 = df3.reset_index()
    elif i == 3:
        df4 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=18
        while h<=23:
            df4.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df4['Value']='DECR4'
        df4 = df4.reset_index()
    elif i == 4:
        df5 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=24

        while h<=29:
            df5.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df5['Value']='DECR5'
        df5 = df5.reset_index()
    elif i == 5:
        df10 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=30
        while h<=35:
            df10.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df10['Value']='DECR10'
        df10 = df10.reset_index()
    elif i == 6:
        df15 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=36
        while h<=41:
            df15.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df15['Value']='DECR15'
        df15 = df15.reset_index()
    elif i == 7:
        df20 = pd.DataFrame(ser[['Total Sum of Revenue','Total Sum of BFTE']],index=ser.index)
        h=42
        while h<=48:
            df20.loc[answersreg1[h][0]]=answersreg1[h][1:]
            h=h+1
        df20['Value']='DECR20' 
        df20 = df20.reset_index()
    else:
        print('It didnt work')
        
ASGneg = pd.concat([df1,df2,df3,df4,df5,df10,df15,df20])

ASGINC=pd.concat([ASGpos,ASGneg])
ASGINC=ASGINC.rename(columns={'Unnamed: 0':'Date'})


from io import StringIO,BytesIO
from azure.storage.blob import BlockBlobService

blobservice = BlockBlobService(account_name='flaskstorage', account_key='M9Hax/c6wKCdVXIcmBafad35/ctWW2OQJQynRMrM29D+mfZXWW53MF0Sthsf0cmWN+/XukVg/aZQ/6XBAB4cgg==') 
data= BytesIO()
ASGINC.to_excel(data, index=False)
data=bytes(data.getvalue())
data=BytesIO(data)
blobservice.create_blob_from_stream('mtscontainer','MTSINC.xlsx',data)
data.close()
"""
from pandas import ExcelWriter
if not os.path.exists('Generated-Excels'):
    os.makedirs('Generated-Excels')
writer = ExcelWriter('Generated-Excels/MTSINCR.xlsx')
ASGINC.to_excel(writer,'Sheet2')
writer.save()
print('done')

"""
























