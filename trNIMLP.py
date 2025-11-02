#'#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nbeats_keras.model import NBeatsNet as NBeatsKeras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
from keras.optimizers import RMSprop, Adam
import time
from keras.models import load_model
#from target_data_electronic70_7 import target_X, target_y ,test_X, test_y
#from m4databasis21_7 import base_domain,zt_in,zt_out,M4Meta,inputsize,train_12,train_12_y
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from tensorflow.keras.losses import Loss
import tensorflow as tf
#from m4databasis35_7_70_7 import train_35,train_35_y,train_70,train_70_y
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add, Concatenate,Flatten,Reshape
#import pandas as pd
import os

#####################################################################################
model_num = int(input('Model k per loss'))
data = input('DATASET')
learning_late = float(input('learning_late'))



target_X= pd.read_csv(f"../data/{data}/train_input_7.csv").iloc[:,1:].values.astype(np.float32) 
target_y =pd.read_csv(f"../data/{data}/train_output_7.csv").iloc[:,1:].values.astype(np.float32) 

X_train = target_X[:-round(target_X.shape[0]*0.2),:].astype(np.float32)
y_train = target_y[:-round(target_y.shape[0]*0.2)].astype(np.float32)
#target_X_val, target_y_val
target_X_val= target_X[-round(target_X.shape[0]*0.2):,:].astype(np.float32)
target_y_val =target_y[-round(target_y.shape[0]*0.2):].astype(np.float32)


test_X= pd.read_csv(f"../data/{data}/val_input_7.csv").iloc[:,1:].values.astype(np.float32) 
test_y =pd.read_csv(f"../data/{data}/val_output_7.csv").iloc[:,1:].values.astype(np.float32) 
#target_X.shape[0]*20
np.random.seed(2)
random_indices1 = np.random.choice(pd.read_csv("../data/M4_train.csv").iloc[:,(1):].index, size=target_X.shape[0]*20, replace=False)
X_train = pd.read_csv("../data/M4_train.csv").iloc[:,1+(24*0):].loc[random_indices1].values
y_train = pd.read_csv("../data/M4_test.csv").iloc[:,1:].loc[random_indices1].values
X_train.shape[1], y_train.shape[1]

target_X.shape,test_X.shape

###################################################
# loss SMAPE
class SMAPE(Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, tf.shape(y_true))  # 예측 값의 차원을 맞춤
       # y_pred=tf.clip_by_value(y_pred, 1e-10, tf.reduce_max(y_pred))
       # y_true = tf.clip_by_value(y_true, 1e-10, tf.reduce_max(y_true))
        
        numerator = 100 * tf.abs(y_true- y_pred )
        denominator =  (tf.abs(y_true ) + tf.abs(y_pred))/2
        smape =  numerator /  denominator #tf.clip_by_value(denominator, 1e-10, tf.reduce_max(denominator))
        return tf.reduce_mean(smape)

#################################################################################
# loss MASE
class MASE(Loss):
    def __init__(self, training_data, period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.calculate_scale(training_data, period)
    def seasonal_diff(data, period):
        return data[period:] - data[:-period]

    def calculate_scale(self, training_data, period):
        # 주기 차분 계산
        diff = seasonal_diff(training_data, period)
        scale = np.mean(np.abs(diff))
        return scale
    
    def call(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, tf.shape(y_true))  # 차원 맞추기
        error = tf.abs(y_true - y_pred)
        return tf.reduce_mean(error / self.scale)

def seasonal_diff(data, period):
    return data[period:] - data[:-period]

#################################################################################
#################################################################################
# 하이퍼파라미터 인자 설정
def hyperparameter():
    # 1 backcast
    # 2 forecast
    # 3 inputdim
    # 4 outputdim
    # 5 unit
    # 6 bacth size
    return X_train.shape[1], y_train.shape[1],1,1,256
#################################################################################
# nbeats + I모델 생성 함수
def bulid_model(backcast_,forecast_,input_dim,output_dim,unit):
    model= NBeatsKeras(backcast_length=backcast_, 
                       forecast_length=forecast_,
                       input_dim=input_dim,
                       output_dim=output_dim,
                       stack_types=(NBeatsKeras.TREND_BLOCK,
                                    NBeatsKeras.TREND_BLOCK,
                                    
                                    NBeatsKeras.TREND_BLOCK,
                                   NBeatsKeras.SEASONALITY_BLOCK,
                                   NBeatsKeras.SEASONALITY_BLOCK,
                                   NBeatsKeras.SEASONALITY_BLOCK)
                   ,nb_blocks_per_stack=1, thetas_dim=(1,2,3,24,12,6),
                   share_weights_in_stack=True, hidden_layer_units=unit)
    return model 
    
#################################################################################
# nbeats + G모델 생성 함수    
def bulid_model_G(backcast_,forecast_,input_dim,output_dim,unit):
    model= NBeatsKeras(backcast_length=backcast_, 
                       forecast_length=forecast_,
                       input_dim=input_dim,
                       output_dim=output_dim,
                       stack_types=(NBeatsKeras.GENERIC_BLOCK,NBeatsKeras.GENERIC_BLOCK)
                   ,nb_blocks_per_stack=5, thetas_dim=(4,4),
                   share_weights_in_stack=False, hidden_layer_units=unit)
    return model 
#################################################################################
# nbeats + I모델 부트스트랩 샘플링 배깅

def train_bagging_models(num_models, loss_fn, epochs_, patience_, batch_size_, lr, save_dir='saved_models'):
    models = {}
    backcast, forecast, in_dim, out_dim, unit = hyperparameter()
    historys = []
    
    # 저장 디렉토리 없으면 생성
    os.makedirs(save_dir, exist_ok=True)
    
    for n in range(num_models):
        K.clear_session()
        model = bulid_model(backcast, forecast, in_dim, out_dim, unit)
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=loss_fn)
        
        # 부트스트랩 샘플링
        select = np.random.choice(len(X_train), size=len(X_train), replace=False)
        X_bootstrap = X_train[select]
        y_bootstrap = y_train[select]

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience_,
            verbose=0,
            restore_best_weights=True
        )

        history = model.fit(
            X_bootstrap, y_bootstrap,
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=0,
            callbacks=[early_stop],
            validation_split=0.2
        )

        # 모델 저장
        model_path = os.path.join(save_dir, f'model_{loss_fn}.h5')
        model.save(model_path)

        models[f'model_{n+1}'] = model
        historys.append(history)
        print(f"######################################################## Model {n+1} 저장 완료")
    
    return models, historys
#################################################################################
# nbeats + I모델 부트스트랩 샘플링 배깅

def train_bagging_models_G(num_models, loss_fn , epochs_, patience_,batch_size_,lr):
    models = {}
    backcast,forecast,in_dim,out_dim,unit = hyperparameter()
    historys = []
    for n in range(num_models):
        K.clear_session()
        model = bulid_model_G(backcast,forecast,in_dim,out_dim,unit)
       # model.set_weights(pretrained_weights)  # 전이 학습 가중치 적용
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer , loss=loss_fn)
        
        # 부트스트랩 샘플링
        select = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_bootstrap = X_train[select]
        y_bootstrap = y_train[select]
        early_stop = EarlyStopping(monitor='val_loss', patience = patience_, restore_best_weights=True)
        history = model.fit(X_bootstrap, y_bootstrap, batch_size = batch_size_,
                  epochs=epochs_, verbose=0, 
                  callbacks=[early_stop],
                 validation_split = 0.2)
        models[f'model_{n+1}'] = model
        historys.append(history)
        #models.append(model)
        print(f"'########################################################Model{n}")
    return models,historys

#################################################################################
##########################################################################################
# 트랜스퍼 레이어
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = layers.Dropout(rate=dropout)

        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]

        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        x = x + self.pe[:, :tf.shape(x)[1], :]
        return self.dropout(x)
##########################################################################################
# 트랜스퍼 레이어
def create_model(fn,d_model, nlayers, nhead, dropout, iw, ow,lr,pretrained_output_reshaped,inputs):
    
    
    x = layers.Dense(d_model // 2, activation='relu')(pretrained_output_reshaped)
    x = layers.Dense(d_model, activation='relu')(x)
    
    pos_encoding = PositionalEncoding(d_model, dropout)
    x = pos_encoding(x)
    
    for _ in range(nlayers):
        attn_output = layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = layers.Dense(d_model, activation='relu')(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    x = layers.Dense(d_model // 2, activation='relu')(x)
    x = layers.Dense(1)(x)
    x = tf.squeeze(x, axis=-1)
    
    outputs = layers.Dense((iw + ow) // 2, activation='relu')(x)
    outputs = layers.Dense(ow)(outputs)
    
    optimizer = Adam(learning_rate=lr)
    target_model = Model(inputs=inputs, outputs=outputs)
    target_model.compile(optimizer=optimizer, loss=fn)
    
    return target_model
########################################################################################################################
def transfer_FC(model_num, trainable, lossf, epochs_, batch_size_, pt, lr_, model_path='saved_models',modelloss = 'mae'):
    history_mapes_G = []
    model_pred_val = []
    model_pred_test = []
    
    for i in range(1, model_num + 1):
        K.clear_session()
        model_file = os.path.join(model_path, modelloss)
        model1 = load_model(model_file)
        
        # 기존 모델의 마지막 레이어 제외, 나머지 동결 여부 설정
        for layer in model1.layers[:-1]:
            layer.trainable = trainable

        # 전이용 모델 구성
        pretrained_layers = model1.layers[:-1]
        pretrained_model = Model(inputs=model1.input, outputs=pretrained_layers[-1].output)

        # 새로운 입력 (타겟 도메인 시계열 입력)
        inputs = Input(shape=(target_X.shape[1], 1))
        pretrained_output = pretrained_model(inputs)
        pretrained_output_reshaped = layers.Reshape((target_y.shape[1], -1))(pretrained_output)
        
        # FC 추가
        x = layers.Dense(128, activation='linear',
                 kernel_regularizer=regularizers.l1(1e-5))(pretrained_output_reshaped)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='linear')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='linear')(x)
        
        model_instance = Model(inputs=inputs, outputs=output)
        optimizer = Adam(learning_rate=lr_)
        model_instance.compile(optimizer=optimizer, loss=lossf)

        early_stop = EarlyStopping(monitor='val_loss', patience=pt, verbose=0, restore_best_weights=True)

        history = model_instance.fit(
            target_X, target_y,
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=0,
            callbacks=[early_stop],
            validation_data=(target_X_val, target_y_val)
        )

        pred_val = model_instance.predict(target_X_val).reshape(-1, target_y.shape[1])
        pred_test = model_instance.predict(test_X).reshape(-1, target_y.shape[1])

        model_pred_val.append(pred_val)
        model_pred_test.append(pred_test)
        history_mapes_G.append(history)
        
        print(f"######################################################## Loaded and fine-tuned model {i}")
        #del model_instance
    return model_pred_val, model_pred_test
#################################################################################
def transfer_FC_SMAPE(model_num, trainable, lossf, epochs_, batch_size_, pt, lr_, model_path='saved_smape',modelloss = 'mae'):
    history_mapes_G = []
    model_pred_val = []
    model_pred_test = []
    
    for i in range(1, model_num + 1):
        K.clear_session()
        model_file = os.path.join(model_path, f'pretrain_smape.h5')
        # custom_objects 인자를 추가하여 SMAPE를 등록
        model1 = load_model(model_file, custom_objects={'SMAPE': SMAPE})
        
        # 기존 모델의 마지막 레이어 제외, 나머지 동결 여부 설정
        for layer in model1.layers[:-1]:
            layer.trainable = trainable

        pretrained_layers = model1.layers[:-1]
        pretrained_model = Model(inputs=model1.input, outputs=pretrained_layers[-1].output)

        # 새로운 입력 설정 (예: target_X.shape[1]과 target_y.shape[1]에 맞게)
        inputs = Input(shape=(target_X.shape[1], 1))
        pretrained_output = pretrained_model(inputs)
        pretrained_output_reshaped = layers.Reshape((target_y.shape[1], -1))(pretrained_output)
        
        # 추가 FC 레이어 구성
        x = layers.Dense(128, activation='linear')(pretrained_output_reshaped)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='linear')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='linear')(x)
        
        model_instance = Model(inputs=inputs, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_)
        model_instance.compile(optimizer=optimizer, loss=lossf)

        early_stop = EarlyStopping(monitor='val_loss', patience=pt, verbose=0, restore_best_weights=True)

        history = model_instance.fit(
            target_X, target_y,
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=0,
            callbacks=[early_stop],
            validation_data=(target_X_val, target_y_val)
        )

        pred_val = model_instance.predict(target_X_val).reshape(-1, target_y.shape[1])
        pred_test = model_instance.predict(test_X).reshape(-1, target_y.shape[1])

        model_pred_val.append(pred_val)
        model_pred_test.append(pred_test)
        history_mapes_G.append(history)
        
        print(f"######################################################## Loaded and fine-tuned model {i}")
        #del model_instance
    return model_pred_val, model_pred_test

################################################################################
def transfer_FC_MASE(model_num, models, trainable, lossf, epochs_, batch_size_, pt, lr_):
    history_mapes_G = []
    model_pred_val = []
    model_pred_test = []
    for i in range(1, model_num + 1):
        K.clear_session()
        model_name = f'model_{i}'
        m, _ = models
        model1 = m[model_name]
        
        # 모든 레이어를 학습 불가능하게 설정
        for layer in model1.layers[:-1]:
            layer.trainable = trainable
            
        pretrained_layers = model1.layers[:-1]
        pretrained_model = Model(inputs=model1.input, outputs=pretrained_layers[-1].output)

        # LSTM 입력을 위한 입력 레이어
        inputs = Input(shape=(target_X.shape[1], 1))  # X_train의 shape에 맞게 설정
        pretrained_output = pretrained_model(inputs)
        pretrained_output_reshaped = layers.Reshape((y_train.shape[1], -1))(pretrained_output)
        # LSTM 레이어 추가
        lstm_output = layers.Dense(128, activation='linear')(pretrained_output_reshaped ) 
        lstm_output = layers.Dropout(0.2)(lstm_output)
        lstm_output = layers.Dense(64, activation='linear')(lstm_output)# LSTM 레이어 추가
        lstm_output = layers.Dropout(0.2)(lstm_output)
        #lstm_output = layers.Dense(32, activation='linear')(lstm_output)# LSTM 레이어 추가
        #lstm_output = layers.Dropout(0.2)(lstm_output)
       #lstm_output = layers.Dense(y_train.shape[1], activation='linear')(lstm_output)  # 최종 출력 레이어
        #lstm_output = layers.Lambda(lambda x: x[:, -24:, :])(lstm_output)

        lstm_output = layers.Dense(1, activation='linear')(lstm_output)
        model_instance = Model(inputs=inputs, outputs=lstm_output)

        model_instance.compile(optimizer='adam', loss=lossf)  # 모델 컴파일
        early_stop = EarlyStopping(monitor='val_loss', patience=pt, verbose=0, restore_best_weights=True)
    
        # 모델 학습
        history = model_instance.fit(target_X, target_y, batch_size=batch_size_,
                                      epochs=epochs_, verbose=0, 
                                      callbacks=[early_stop],
                                      validation_data=[target_X_val,target_y_val])
        
        # 예측
        pred_val = model_instance.predict(target_X_val)
        pred_val = pred_val.reshape(-1, y_train.shape[1])
        model_pred_val.append(pred_val)
        
        pred_test = model_instance.predict(test_X)
        pred_test = pred_test.reshape(-1, y_train.shape[1])
        model_pred_test.append(pred_test)

        history_mapes_G.append(history)
        print(f"########################################################fitted {i}")

        #del model_instance
    return model_pred_val,model_pred_test
# 예측

def bagging_predict(models, X):
    predictions = np.array([model.predict(X) for model in models.values()])
    return np.median(predictions, axis=0)

def bagging_predict2(models, X):
    predictions = np.array([model.predict(X) for model in models.values()])
    return predictions


# In[46]:


mase_models=train_bagging_models(model_num,MASE(y_train,y_train.shape[1]),2000,10,256,0.001)
mase_pred ,mase_pred2 = transfer_FC_MASE(model_num,mase_models,True, MASE(target_y,y_train.shape[1]),2000,8,10,learning_late)


# In[47]:


mae_pred, mae_pred2 = transfer_FC(
    model_num=model_num,
    trainable=True,
    lossf='mae',
    epochs_=2000,
    batch_size_=8,
    pt=10,
    lr_=learning_late,
    model_path='../pretrain/pretrain_NI/',
    modelloss = 'pretrain_mae.h5'  # 생략 가능
)

mse_pred, mse_pred2 = transfer_FC(
    model_num=model_num,
    trainable=True,
    lossf='mse',
    epochs_=2000,
    batch_size_=8,
    pt=10,
    lr_=learning_late,
    model_path='../pretrain/pretrain_NI/',
    modelloss = 'pretrain_mse.h5'  # 생략 가능
)

mape_pred, mape_pred2 = transfer_FC(
    model_num=model_num,
    trainable=True,
    lossf='mape',
    epochs_=2000,
    batch_size_=8,
    pt=10,
    lr_=learning_late,
    model_path='../pretrain/pretrain_NI/',
    modelloss = 'pretrain_mape.h5'  # 생략 가능
)

smape_pred, smape_pred2 = transfer_FC_SMAPE(
    model_num=model_num,
    trainable=True,
    lossf=SMAPE(),
    epochs_=2000,
    batch_size_=8,
    pt=10,
    lr_=learning_late,
    model_path='../pretrain/pretrain_NI/',
    modelloss = 'pretrain_smape.h5'  # 생략 가능
)


# In[54]:





# In[ ]:
pd.DataFrame(np.array(mse_pred).reshape(1,-1)).to_csv(f'result/val/trNI_{data}_mse_pred.csv')
pd.DataFrame(np.array(mae_pred).reshape(1,-1)).to_csv(f'result/val/trNI_{data}_mae_pred.csv')
pd.DataFrame(np.array(mape_pred).reshape(1,-1)).to_csv(f'result/val/trNI_{data}_mape_pred.csv')
pd.DataFrame(np.array(smape_pred).reshape(1,-1)).to_csv(f'result/val/trNI_{data}_smape_pred.csv')
pd.DataFrame(np.array(mase_pred).reshape(1,-1)).to_csv(f'result/val/trNI_{data}_mase_pred.csv')

#name = 'NBEATs_I+FC'

print("##############################")

concat_G = np.concatenate([mape_pred,mase_pred,smape_pred,mse_pred,mae_pred])
fin_pred_G = np.median(concat_G,axis=0)
print('all',np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([mae_pred, mase_pred,mse_pred])
fin_pred_G = np.median(concat_G,axis=0)
print('best',np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.nan_to_num(np.array(mse_pred), nan=0)])
fin_pred_G = np.median(concat_G,axis=0)
print('mse',np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.array(mase_pred)])
fin_pred_G = np.median(concat_G,axis=0)
print('mase',np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.nan_to_num(np.array(mae_pred), nan=0)])
fin_pred_G = np.median(concat_G,axis=0)
print('mae',np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.array(mape_pred)])
fin_pred_G = np.median(concat_G,axis=0)
print('mape',np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.array(smape_pred)])
fin_pred_G = np.median(concat_G,axis=0)
print('mape',np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_G.flatten())).round(5))

concat_mase = np.concatenate([np.nan_to_num(np.array(mase_pred), nan=0)])
fin_pred_mase = np.median(concat_mase,axis=0)
MASE= np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_mase .flatten())).round(5)

concat_mape = np.concatenate([np.nan_to_num(np.array(mape_pred), nan=0)])
fin_pred_mape = np.median(concat_mape,axis=0)
MAPE= np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_mape .flatten())).round(5)

concat_smape = np.concatenate([np.nan_to_num(np.array(smape_pred), nan=0)])
fin_pred_smape = np.median(concat_smape,axis=0)
sMAPE= np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_smape .flatten())).round(5)

concat_mae = np.concatenate([np.nan_to_num(np.array(mae_pred), nan=0)])
fin_pred_mae = np.median(concat_mae,axis=0)
MAE= np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_mae .flatten())).round(5)

concat_mse = np.concatenate([np.nan_to_num(np.array(mse_pred), nan=0)])
fin_pred_mse = np.median(concat_mse,axis=0)
MSE= np.sqrt(mean_squared_error(target_y_val.flatten(),fin_pred_mse .flatten())).round(5)

performance = np.array([MASE, MAPE,sMAPE,MAE,MSE])

pd.DataFrame(performance).to_csv(f"result/trNINLP_{data}_weight.csv")


# In[31]:


pd.DataFrame(np.array(mse_pred2).reshape(1,-1)).to_csv(f'result/test/trNI_{data}_mse_pred.csv')
pd.DataFrame(np.array(mae_pred2).reshape(1,-1)).to_csv(f'result/test/trNI_{data}_mae_pred.csv')
pd.DataFrame(np.array(mape_pred2).reshape(1,-1)).to_csv(f'result/test/trNI_{data}_mape_pred.csv')
pd.DataFrame(np.array(smape_pred2).reshape(1,-1)).to_csv(f'result/test/trNI_{data}_smape_pred.csv')
pd.DataFrame(np.array(mase_pred2).reshape(1,-1)).to_csv(f'result/test/trNI_{data}_mase_pred.csv')

