import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
df= pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
df=df.iloc[:,6:]
df=df.values

def remove_0_period(data):
    b=[]
    for k in range(len(data)):
        for i in range(len(data[k])):
            if int(data[k][i]) != 0: 
                b.append(df[k][i:])
                break
    return b
def check_0_series(array):
    count =0
    for i in array:
        if i== 0: count = count +1
    if count> len(array)* 0.7: return True
    else: return False
    
def mean_std(dt):
    mean=[]
    std=[]
    for i in range(len(dt)):
        mean.append(dt[i].mean())
        std.append(dt[i].std())
    return np.array(mean),np.array(std)

STEP=28
FUTURE_STEP=28
HISTORY_STEP = 365
MARK = len(df[0]) - (FUTURE_STEP+HISTORY_STEP+STEP)
train_split= int(MARK*0.8) +FUTURE_STEP+HISTORY_STEP+STEP
val_split = len(df[0]) - int(MARK*0.2) - (FUTURE_STEP+HISTORY_STEP+STEP)

def univariate_data(dt,past_step, future_step,step):
    output =[]
    label = []
   
    for j in range(0, len(dt)):
        dataset = dt[j]
        
        for i in range(past_step,(len(dataset)-future_step),step):
           # indices= np.reshape(dataset[(i-past_step):i],(past_step,1))
            indices= np.reshape(dataset[(i-past_step):i],(past_step,1))                      
            
            if  check_0_series(indices)==False:
                indices = (indices-mean[j])/std[j]
                output.append(indices)
                label.append((dataset[(i+step):(i+future_step+step)]-mean[j])/std[j])
    return np.array(output), label
 
### Data split
  x1_train,y_train = univariate_data(df_train, HISTORY_STEP,FUTURE_STEP,STEP)
  x1_val,y_val =  univariate_data(df_val, HISTORY_STEP,FUTURE_STEP,STEP)
  
 ### Model
EVALUATION_INTERVAL = 200
EPOCHS = 10
model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x1_val.shape[-2:]))
model1.add(tf.keras.layers.LSTM(16, activation='relu'))
model1.add(tf.keras.layers.Dense(28))

model1.compile(optimizer='adam', loss='mse')
history = model1.fit(train_data, epochs=10,steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data,validation_steps=50)
