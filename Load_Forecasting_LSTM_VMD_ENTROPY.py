import datetime
an = datetime.datetime.now()

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense,Dropout,LSTM, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from keras.models import Sequential
from vmdpy import VMD
from pyentrp import entropy as ent
import matplotlib.pyplot as plt


df = pd.read_excel('energy_consumption.xlsx')

time_index = pd.date_range(start='2017-01-01 00:00', periods=len(df),  freq='H')  
time_index = pd.DatetimeIndex(time_index)
df['Time']=time_index
df.index=df["Time"]
df.drop(["Time"], axis=1, inplace=True)
df=np.array(df)


test_size=0.1
Epochs=100
Batch_size=32
Dropout_rate=0.2
lookback=24
activation='tanh'
learning_rate=1e-3
initial_learning_rate = 0.1
validation_split=0.15
los="mean_squared_error"

alpha = 10000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 11              # 3 modes  
DC = 0             # no DC part imposed  
init = 2           # initialize omegas uniformly  
tol = 1e-6  


# Run VMD 
u, u_hat, omega = VMD(df, alpha, tau, K, DC, init, tol)

df=pd.DataFrame(u.T) 

#-------calculation of the entropy value of each IMF separately---------
pe=[]
for i in range(0,df.shape[1]):
    pe.append(ent.permutation_entropy(df[i], order=3, normalize=True))
    
dt=pd.DataFrame(np.zeros((len(df),6)))

dt[0]=df[1]+df[7]
dt[1]=df[5]
dt[2]=df[2]+df[3]+df[4]+df[6]
dt[3]=df[0]
dt[4]=df[9]+df[8]
dt[5]=df[10]


test_size=0.1
Epochs=100
Batch_size=32
Dropout_rate=0.2
lookback=24
activation='tanh'
learning_rate=1e-3
validation_split=0.15
los="mean_squared_error"

#----------------------Train and test split--------------------
def split_data(dataframe, test_size):
    pos=int(round(len(dataframe)*(1-test_size)))
    train=dataframe[:pos]
    test=dataframe[pos:]
    return train,test,pos

def create_feature(data, lookback):
    X, y=[], []
    for i in range(lookback, len(data)):
        X.append([data[i-lookback:i,0]])
        y.append(data[i,0])
    return np.array(X), np.array(y)

scaler=MinMaxScaler(feature_range=(0,1))
#-------------------------------Building the model-------------------------

def create_model(df):
    train,test,pos=split_data(df,test_size)
    train=scaler.fit_transform(train)
    test=scaler.fit_transform(test)
    X_train, y_train=create_feature(train,lookback)
    X_test, y_test=create_feature(test,lookback)
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(X_train.shape[1],lookback)))
    lstm_model.add(LSTM(units=256,activation=activation, return_sequences=True))
    lstm_model.add(Dropout(Dropout_rate))
    lstm_model.add(LSTM(units=128,activation=activation, return_sequences=True))
    lstm_model.add(Dropout(Dropout_rate))
    lstm_model.add(LSTM(64,activation='Softmax', return_sequences=False))
    lstm_model.add(Dropout(Dropout_rate))
    lstm_model.add(Dense(1))
    lstm_model.summary()
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=los)
   

    # ---------------------------Train model-------------------------------
    history=lstm_model.fit(X_train, y_train, epochs=Epochs, batch_size=Batch_size, 
                            validation_split=validation_split)

    loss=lstm_model.evaluate(X_test, y_test, batch_size=50)
    print("\nTest Loss=%0.2f%%" % (100*loss))
    #-----------------------------Error Rate Graph---------------------------
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r', label='Eğitim Veri Seti')
    plt.plot(epochs, val_loss, 'b', label='Doğrulama Veri Seti')
    plt.title('Eğitim ve Doğrulama Kayıpları')
    plt.legend(loc=0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
#-----------------------------------Predict------------------------------

    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions=scaler.inverse_transform(lstm_predictions)
    y_test=scaler.inverse_transform(y_test)
    return lstm_predictions, y_test
    
    
#-------------------------------Model evaluate-------------------------

def hata(y_test, lstm_predictions):
    MAE=mean_absolute_error(y_test, lstm_predictions)
    MSE=mean_squared_error(y_test, lstm_predictions)
    MedAE=median_absolute_error(y_test, lstm_predictions)
    r2=r2_score(y_test, lstm_predictions)
    RMSE=np.sqrt(mean_squared_error(y_test, lstm_predictions))
    MAPE=(mean_absolute_percentage_error(y_test, lstm_predictions)*100)
    print("MAE={:.4f}".format(MAE))
    print("MSE={:.4f}".format(MSE))
    print("MedAE= {:.4f}".format(MedAE))
    print("Belirleme Katsayısı(R^2)={:.4f}".format(r2))
    print("RMSE= {:.4f}".format(RMSE))
    print("MAPE= {:.4f}".format(MAPE))
    return MAE, MSE, MedAE, r2, RMSE, MAPE

#------------------------Modelling of each PFs separately--------------------
pos=int(round(len(dt)*(1-test_size)))
imf_predictions=pd.DataFrame(np.zeros(((len(dt)-pos-lookback),dt.shape[1])))
imf_test=pd.DataFrame(np.zeros(((len(dt)-pos-lookback),dt.shape[1])))

for i in range(0,(dt.shape[1])):
    print('\n{}. PFs Eğitimi\n'.format(i))
    data=pd.DataFrame((dt.iloc[:,i].values).reshape(-1, 1))
    imf_predictions[i], imf_test[i]=create_model(data)
    
toplam_predict=np.sum(imf_predictions, axis=1)
toplam_test=np.sum(imf_test, axis=1)

MAE, MSE, MedAE, r2, RMSE, MAPE=hata(toplam_test, toplam_predict)

plt.figure(figsize=(20,6))
plt.plot(time_index[-383:], toplam_predict[-383:], 'r', label='Tahmin Tüketim Değerleri[MWh]')
plt.plot(time_index[-383:], toplam_test[-383:], 'b', label='Gerçek Tüketim Değerleri[MWh]')
plt.title('Gerçek ve Tahmin Tüketim Değerleri(Model-7)')
plt.legend(loc=0)
plt.xlabel('Time')
plt.ylabel('Tüketim[MWh]')
plt.show()


an1=datetime.datetime.now()
print("Total Time:",(an1-an))