#'#!/usr/bin/env python
# coding: utf-8

import os, time, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from nbeats_keras.model import NBeatsNet as NBeatsKeras

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import regularizers
from keras.optimizers import Adam
from keras.models import load_model

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.losses import Loss
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention, Dropout,
    Add, Concatenate, Flatten, Reshape, TimeDistributed, Lambda
)
from tensorflow.keras.utils import custom_object_scope

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

strategy = tf.distribute.MirroredStrategy()
print("Num GPUs Available:", strategy.num_replicas_in_sync)
#####################################################################################
model_num = int(input('Model k per loss'))
data = input('DATASET')             
learning_late = float(input('learning_late'))

LOG_DIR = os.path.join('logstf', data)
os.makedirs(LOG_DIR, exist_ok=True)

## target domain
target_X = pd.read_csv(f"../data/{data}/train_input_7.csv").iloc[:, 1:].values.astype(np.float32)
target_y = pd.read_csv(f"../data/{data}/train_output_7.csv").iloc[:, 1:].values.astype(np.float32)

## 왜 있는 거임???
X_train = target_X[:-round(target_X.shape[0] * 0.2), :].astype(np.float32)
y_train = target_y[:-round(target_y.shape[0] * 0.2)].astype(np.float32)
target_X_val = target_X[-round(target_X.shape[0] * 0.2):, :].astype(np.float32)
target_y_val = target_y[-round(target_y.shape[0] * 0.2):].astype(np.float32)

test_X  = pd.read_csv(f"../data/{data}/val_input_7.csv").iloc[:, 1:].values.astype(np.float32)
test_y  = pd.read_csv(f"../data/{data}/val_output_7.csv").iloc[:, 1:].values.astype(np.float32)

## source domain
np.random.seed(2)
random_indices1 = np.random.choice(pd.read_csv("../data/M4_train.csv").iloc[:, (1):].index,
                                   size=target_X.shape[0] * 20, replace=True)
X_train = pd.read_csv("../data/M4_train.csv").iloc[:, 1 + (24 * 0):].loc[random_indices1].values.astype(np.float32)
y_train = pd.read_csv("../data/M4_test.csv").iloc[:, 1:].loc[random_indices1].values.astype(np.float32)

_ = (X_train.shape[1], y_train.shape[1])
_ = (target_X.shape, test_X.shape)

class SavePredsAndTruthCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, filepath):
        super().__init__()
        self.x_val = validation_data[0]
        self.y_val = validation_data[1] 
        self.filepath = filepath
        self.best_val_loss = float('inf')
        self.best_predictions = None

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_predictions = self.model.predict(self.x_val, verbose=0)

    def on_train_end(self, logs=None):
        if self.best_predictions is not None:
            print(f"\nSaving best validation predictions and ground truth to {self.filepath}")

            horizon = self.y_val.shape[1]
            combined_df = pd.DataFrame()
            
            for i in range(horizon):
                combined_df[f'prediction_{i+1}'] = self.best_predictions[:, i]
                combined_df[f'ground_truth_{i+1}'] = self.y_val[:, i]
            
            combined_df.to_csv(self.filepath, index=False)
        else:
            print("\nWarning: No validation predictions were saved.")
###################################################
class PrintValLossEveryN(tf.keras.callbacks.Callback):
    def __init__(self, n=1):
        super().__init__()
        self.n = n
        self.best_val = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if val_loss is not None and loss is not None:
            # n 단위로 찍기 (train loss + val loss)
            if (epoch + 1) % self.n == 0:
                print(f"Epoch {epoch+1}: loss={loss:.4f}, val_loss={val_loss:.4f}")
            # best val_loss 갱신
            if val_loss < self.best_val:
                self.best_val = val_loss

    def on_train_end(self, logs=None):
        print(f"Training finished. Best val_loss={self.best_val:.4f}")
        
###################################################
# loss SMAPE
class SMAPE(Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, tf.shape(y_true))
        numerator = 100 * tf.abs(y_true - y_pred)
        denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2
        smape = numerator / denominator
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
        diff = seasonal_diff(training_data, period)
        scale = np.mean(np.abs(diff))
        return scale

    def call(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, tf.shape(y_true))
        error = tf.abs(y_true - y_pred)
        return tf.reduce_mean(error / self.scale)

def seasonal_diff(data, period):
    return data[period:] - data[:-period]

#################################################################################
def hyperparameter():
    return X_train.shape[1], y_train.shape[1], 1, 1, 256    ## backcast, forecast, in_dim, out_dim, unit

#################################################################################
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, dropout=0.05, max_len=5000, **kwargs):
        super().__init__(**kwargs)
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
def create_model(fn,d_model, nlayers, nhead, dropout, iw, ow,lr,pretrained_output_reshaped,inputs):
    """
    타겟 모델 구성 과정, 사전 학습 모델 앞에 encoder head를 장착...
    """
    
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
    x = layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(x)
    
    outputs = layers.Dense((iw + ow) // 2, activation='relu')(x)
    outputs = layers.Dense(ow)(outputs)
    
    optimizer = Adam(learning_rate=lr)
    target_model = Model(inputs=inputs, outputs=outputs)    ## pretrained_model에 encoder 헤드를 붙이는 형태 같음
    target_model.compile(optimizer=optimizer, loss=fn)
    
    return target_model
########################################################################################################################

#################################################################################
# ===== 백본: Transformer Encoder =====
def build_transformer_encoder(backcast_len, forecast_len,
                              d_model=512, d_ff=2048, nhead=8,
                              e_layers=2, d_layers=0, dropout=0.05):
    key_dim = d_model // nhead

    inp_flat = Input(shape=(backcast_len,), name='input_flat')
    inp = Reshape((backcast_len, 1), name='input_reshaped')(inp_flat)

    enc = Dense(d_model, activation='linear', name='enc_proj')(inp)
    enc = PositionalEncoding(d_model, dropout, max_len=backcast_len)(enc)

    for li in range(e_layers):
        attn_out = MultiHeadAttention(
            num_heads=nhead, key_dim=key_dim, dropout=dropout, name=f'enc_mha_{li}'
        )(enc, enc)
        attn_out = Dropout(dropout, name=f'enc_mha_do_{li}')(attn_out)
        enc = Add(name=f'enc_mha_add_{li}')([enc, attn_out])
        enc = LayerNormalization(epsilon=1e-6, name=f'enc_ln1_{li}')(enc)

        ffn_out = Dense(d_ff, activation='relu', name=f'enc_ffn1_{li}')(enc)
        ffn_out = Dense(d_model, activation='linear', name=f'enc_ffn2_{li}')(ffn_out)
        ffn_out = Dropout(dropout, name=f'enc_ffn_do_{li}')(ffn_out)
        enc = Add(name=f'enc_ffn_add_{li}')([enc, ffn_out])
        enc = LayerNormalization(epsilon=1e-6, name=f'enc_ln2_{li}')(enc)

    enc_tail = Lambda(lambda t: t[:, -forecast_len:, :], name='enc_tail')(enc)

    penultimate = Reshape((forecast_len * d_model,), name='penultimate_features')(enc_tail)
    pretrain_out = Dense(forecast_len, activation='linear', name='pretrain_head')(penultimate)

    model = Model(inputs=inp_flat, outputs=pretrain_out, name='TransformerEncoderBackbone')
    return model


#################################################################################
# 백본을 Transformer ED로
def train_bagging_models(num_models, loss_fn, epochs_, patience_, batch_size_, lr, save_dir='saved_models'):
    models = {}
    backcast, forecast, in_dim, out_dim, unit = hyperparameter()
    historys = []
    os.makedirs(save_dir, exist_ok=True)

    for n in range(num_models):
        K.clear_session()
        import gc; gc.collect()

        ## using GPU
        with strategy.scope():
            base = build_transformer_encoder(
                backcast_len=backcast, forecast_len=forecast,
                d_model=512, d_ff=2048, nhead=8, e_layers=2, d_layers=0, dropout=0.05
            )
            base.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn)

        select = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_bootstrap = X_train[select]
        y_bootstrap = y_train[select]

        val_split_index = int(len(X_bootstrap) * 0.8)
        X_train_split, X_val_split = X_bootstrap[:val_split_index], X_bootstrap[val_split_index:]
        y_train_split, y_val_split = y_bootstrap[:val_split_index], y_bootstrap[val_split_index:]

        loss_name = loss_fn if isinstance(loss_fn, str) else loss_fn.__class__.__name__
        
        csv_log = CSVLogger(os.path.join(LOG_DIR, f'pretrain_{loss_name}_model{n+1}.csv'))
        early_stop = EarlyStopping(monitor='val_loss', patience=patience_, verbose=0, restore_best_weights=True)
        
        pred_filepath = os.path.join(LOG_DIR, f'pretrain_val_results_{loss_name}_model{n+1}.csv')
        save_preds_callback = SavePredsAndTruthCallback(
            validation_data=(X_val_split, y_val_split), 
            filepath=pred_filepath
        )
        
        history = base.fit(
            X_train_split, y_train_split,  
            batch_size=batch_size_, 
            epochs=epochs_, 
            verbose=0,
            callbacks=[early_stop, csv_log, PrintValLossEveryN(1), save_preds_callback], 
            validation_data=(X_val_split, y_val_split)
        )
        
        model_path = os.path.join(save_dir, f'model_{loss_name}_{n+1}.h5')
        base.save(model_path)

        models[f'model_{n+1}'] = base
        historys.append(history)
        print(f"######################################################## Transformer-ED Backbone Model {n+1} 저장 완료")

    return models, historys

#################################################################################
def transfer_FC(model_num, trainable, lossf, epochs_, batch_size_, pt, lr_,
                model_path='../pretrain/pretrain_NI/', modelloss='pretrain_mae.h5'):
    history_mapes_G, model_pred_val, model_pred_test = [], [], []

    for i in range(1, model_num + 1):
        K.clear_session()
        import gc; gc.collect()
        model_file = os.path.join('saved_models', f'model_{lossf}_{i}.h5')
        with custom_object_scope({'PositionalEncoding': PositionalEncoding}):
            base_loaded = load_model(model_file, compile=False, safe_mode=False)

        for layer in base_loaded.layers[:-1]:
            layer.trainable = trainable

        pretrained_layers = base_loaded.layers[:-1]
        pretrained_model = Model(inputs=base_loaded.input, outputs=pretrained_layers[-1].output)

        inputs = Input(shape=(target_X.shape[1], 1))
        flat_inp = layers.Reshape((target_X.shape[1],), name='transfer_flatten')(inputs)
        pretrained_output = pretrained_model(flat_inp)                                  
        pretrained_output_reshaped = Reshape((target_y.shape[1], -1))(pretrained_output)

        model_instance = create_model(lossf ,d_model=84, nlayers=4,nhead=4, dropout=0.2,
                                      iw=X_train.shape[1], ow=y_train.shape[1],lr=lr_ ,  
                            pretrained_output_reshaped=pretrained_output_reshaped,inputs=inputs)

        early_stop = EarlyStopping(monitor='val_loss', patience=pt, verbose=0, restore_best_weights=True)
        csv_log   = CSVLogger(os.path.join(LOG_DIR, f'transfer_{str(lossf)}_lr{lr_}_run{i}.csv'))

        history = model_instance.fit(
            target_X.reshape(-1, target_X.shape[1], 1),            
            target_y,                                              
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=0,
            callbacks=[early_stop, csv_log, PrintValLossEveryN(1)],
            validation_data=(
                target_X_val.reshape(-1, target_X_val.shape[1], 1),
                target_y_val                                       
            )
        )

        pred_val = model_instance.predict(target_X_val.reshape(-1, target_X_val.shape[1], 1), verbose=0).reshape(-1, target_y.shape[1])
        pred_test = model_instance.predict(test_X.reshape(-1, test_X.shape[1], 1), verbose=0).reshape(-1, target_y.shape[1])

        model_pred_val.append(pred_val)
        model_pred_test.append(pred_test)
        history_mapes_G.append(history)
        print(f"######################################################## Loaded & fine-tuned (FC) model {i}")
        del model_instance
        del base_loaded
        del pretrained_model
        tf.keras.backend.clear_session()
        import gc; gc.collect()
        
    return model_pred_val, model_pred_test

def transfer_FC_SMAPE(model_num, trainable, lossf, epochs_, batch_size_, pt, lr_,
                      model_path='../pretrain/pretrain_NI/', modelloss='pretrain_smape.h5'):
    history_mapes_G, model_pred_val, model_pred_test = [], [], []

    for i in range(1, model_num + 1):
        K.clear_session()
        import gc; gc.collect()
        model_file = os.path.join('saved_models', f'model_SMAPE_{i}.h5')
        custom_objects = {
            'PositionalEncoding': PositionalEncoding, 
            'SMAPE': SMAPE
        }
        base_loaded = load_model(model_file, custom_objects=custom_objects, compile=False, safe_mode=False)

        for layer in base_loaded.layers[:-1]:
            layer.trainable = trainable
        pretrained_layers = base_loaded.layers[:-1]
        pretrained_model = Model(inputs=base_loaded.input, outputs=pretrained_layers[-1].output)
        

        inputs = Input(shape=(target_X.shape[1], 1))
        flat_inp = layers.Reshape((target_X.shape[1],), name='transfer_flatten')(inputs)
        pretrained_output = pretrained_model(flat_inp)
        pretrained_output_reshaped = Reshape((target_y.shape[1], -1))(pretrained_output)

        model_instance = create_model(lossf ,d_model=84, nlayers=4,nhead=4, dropout=0.2,
                                      iw=X_train.shape[1], ow=y_train.shape[1],lr=lr_ ,  
                            pretrained_output_reshaped=pretrained_output_reshaped,inputs=inputs)

        early_stop = EarlyStopping(monitor='val_loss', patience=pt, verbose=0, restore_best_weights=True)
        csv_log   = CSVLogger(os.path.join(LOG_DIR, f'transfer_SMAPE_lr{lr_}_run{i}.csv'))

        history = model_instance.fit(
            target_X.reshape(-1, target_X.shape[1], 1),             
            target_y,                                               
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=0,
            callbacks=[early_stop, csv_log, PrintValLossEveryN(1)],
            validation_data=(
                target_X_val.reshape(-1, target_X_val.shape[1], 1), 
                target_y_val                                        
            )
        )

        pred_val = model_instance.predict(target_X_val.reshape(-1, target_X_val.shape[1], 1), verbose=0).reshape(-1, target_y.shape[1])
        pred_test = model_instance.predict(test_X.reshape(-1, test_X.shape[1], 1), verbose=0).reshape(-1, target_y.shape[1])

        model_pred_val.append(pred_val)
        model_pred_test.append(pred_test)
        history_mapes_G.append(history)
        print(f"######################################################## Loaded & fine-tuned (SMAPE) model {i}")
        del model_instance
        del base_loaded
        del pretrained_model
        tf.keras.backend.clear_session()
        import gc; gc.collect()
    return model_pred_val, model_pred_test

def transfer_FC_MASE(model_num, trainable, lossf, epochs_, batch_size_, pt, lr_):
    history_mapes_G, model_pred_val, model_pred_test = [], [], []

    for i in range(1, model_num + 1):
        K.clear_session()
        import gc; gc.collect()
        model_file = os.path.join('saved_models', f'model_MASE_{i}.h5')
        custom_objects = {
            'PositionalEncoding': PositionalEncoding, 
            'MASE': lossf
        }
        base_loaded = load_model(model_file, custom_objects=custom_objects, compile=False, safe_mode=False)
        
        for layer in base_loaded.layers[:-1]:
            layer.trainable = trainable
        pretrained_layers = base_loaded.layers[:-1]
        pretrained_model  = Model(inputs=base_loaded.input, outputs=pretrained_layers[-1].output)

        inputs = Input(shape=(target_X.shape[1], 1))
        flat_inp = layers.Reshape((target_X.shape[1],), name='transfer_flatten')(inputs)
        pretrained_output = pretrained_model(flat_inp)                         
        pretrained_output_reshaped = Reshape((y_train.shape[1], -1))(pretrained_output)


        model_instance = create_model(lossf ,d_model=84, nlayers=4,nhead=4, dropout=0.2,
                                          iw=X_train.shape[1], ow=y_train.shape[1],lr=lr_ ,  
                                pretrained_output_reshaped=pretrained_output_reshaped,inputs=inputs)

        early_stop = EarlyStopping(monitor='val_loss', patience=pt, verbose=0, restore_best_weights=True)
        csv_log   = CSVLogger(os.path.join(LOG_DIR, f'transfer_MASE_lr{lr_}_run{i}.csv'))

        history = model_instance.fit(
            target_X.reshape(-1, target_X.shape[1], 1),          
            target_y,                                            
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=0,
            callbacks=[early_stop, csv_log, PrintValLossEveryN(1)],
            validation_data=(
                target_X_val.reshape(-1, target_X_val.shape[1], 1),
                target_y_val                                       
            )
        )

        pred_val = model_instance.predict(target_X_val.reshape(-1, target_X_val.shape[1], 1), verbose=0)
        pred_val = pred_val.reshape(-1, y_train.shape[1])
        model_pred_val.append(pred_val)

        pred_test = model_instance.predict(test_X.reshape(-1, test_X.shape[1], 1), verbose=0)
        pred_test = pred_test.reshape(-1, y_train.shape[1])
        model_pred_test.append(pred_test)

        history_mapes_G.append(history)
        print(f"######################################################## Transformer-ED (MASE) fitted {i}")
        del model_instance
        del base_loaded
        del pretrained_model
        tf.keras.backend.clear_session()
        import gc; gc.collect()
        
    return model_pred_val, model_pred_test

def bagging_predict(models, X):
    predictions = np.array([model.predict(X) for model in models.values()])
    return np.median(predictions, axis=0)

def bagging_predict2(models, X):
    predictions = np.array([model.predict(X) for model in models.values()])
    return predictions


# ================================ 실행부 ================================
lr = 0.0001
print("\n--- Starting MSE Pre-training ---")
_ = train_bagging_models(model_num, 'mse', 2000, 10, 256, lr)

print("\n--- Starting MAE Pre-training ---")
_ = train_bagging_models(model_num, 'mae', 2000, 10, 256, lr) 

print("--- Starting MASE Pre-training ---")
mase_models = train_bagging_models(model_num, MASE(y_train, y_train.shape[1]), 2000, 10, 256, lr)

print("\n--- Starting MAPE Pre-training ---")
_ = train_bagging_models(model_num, 'mape', 2000, 10, 256, lr)

print("\n--- Starting SMAPE Pre-training ---")
_ = train_bagging_models(model_num, SMAPE(), 2000, 10, 256, lr)

os.makedirs('resulttf/val', exist_ok=True)
os.makedirs('resulttf/test', exist_ok=True)

mase_pred, mase_pred2 = transfer_FC_MASE(model_num, True, MASE(target_y, y_train.shape[1]), 2000, 8, 10, learning_late)
pd.DataFrame(np.array(mase_pred).reshape(1, -1)).to_csv(f'resulttf/val/trTFTF_{data}_mase_pred.csv')
pd.DataFrame(np.array(mase_pred2).reshape(1, -1)).to_csv(f'resulttf/test/trTFTF_{data}_mase_pred.csv')

## fine-tuning. not freezing
mae_pred, mae_pred2 = transfer_FC(
    model_num=model_num, trainable=True, lossf='mae',
    epochs_=2000, batch_size_=8, pt=10, lr_=learning_late
)
pd.DataFrame(np.array(mae_pred).reshape(1, -1)).to_csv(f'resulttf/val/trTFTF_{data}_mae_pred.csv')
pd.DataFrame(np.array(mae_pred2).reshape(1, -1)).to_csv(f'resulttf/test/trTFTF_{data}_mae_pred.csv')

mse_pred, mse_pred2 = transfer_FC(
    model_num=model_num, trainable=True, lossf='mse',
    epochs_=2000, batch_size_=8, pt=10, lr_=learning_late
)
pd.DataFrame(np.array(mse_pred).reshape(1, -1)).to_csv(f'resulttf/val/trTFTF_{data}_mse_pred.csv')
pd.DataFrame(np.array(mse_pred2).reshape(1, -1)).to_csv(f'resulttf/test/trTFTF_{data}_mse_pred.csv')

mape_pred, mape_pred2 = transfer_FC(
    model_num=model_num, trainable=True, lossf='mape',
    epochs_=2000, batch_size_=8, pt=10, lr_=learning_late
)
pd.DataFrame(np.array(mape_pred2).reshape(1, -1)).to_csv(f'resulttf/test/trTFTF_{data}_mape_pred.csv')
pd.DataFrame(np.array(mape_pred).reshape(1, -1)).to_csv(f'resulttf/val/trTFTF_{data}_mape_pred.csv')

smape_pred, smape_pred2 = transfer_FC_SMAPE(
    model_num=model_num, trainable=True, lossf=SMAPE(),
    epochs_=2000, batch_size_=8, pt=10, lr_=learning_late
)
pd.DataFrame(np.array(smape_pred).reshape(1, -1)).to_csv(f'resulttf/val/trTFTF_{data}_smape_pred.csv')
pd.DataFrame(np.array(smape_pred2).reshape(1, -1)).to_csv(f'resulttf/test/trTFTF_{data}_smape_pred.csv')

print("##############################")

# 전체/부분 앙상블 RMSE 출력
concat_G = np.concatenate([mape_pred, mase_pred, smape_pred, mse_pred, mae_pred])
fin_pred_G = np.median(concat_G, axis=0)
print('all', np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([mae_pred, mase_pred, mse_pred])
fin_pred_G = np.median(concat_G, axis=0)
print('best', np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.nan_to_num(np.array(mse_pred), nan=0)])
fin_pred_G = np.median(concat_G, axis=0)
print('mse', np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.array(mase_pred)])
fin_pred_G = np.median(concat_G, axis=0)
print('mase', np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.nan_to_num(np.array(mae_pred), nan=0)])
fin_pred_G = np.median(concat_G, axis=0)
print('mae', np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.array(mape_pred)])
fin_pred_G = np.median(concat_G, axis=0)
print('mape', np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))

concat_G = np.concatenate([np.array(smape_pred)])
fin_pred_G = np.median(concat_G, axis=0)
print('smape', np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))

# 지표별 단독 RMSE 저장
concat_mase = np.concatenate([np.nan_to_num(np.array(mase_pred), nan=0)])
fin_pred_mase = np.median(concat_mase, axis=0)
MASE_rmse = np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_mase.flatten())).round(5)

concat_mape = np.concatenate([np.nan_to_num(np.array(mape_pred), nan=0)])
fin_pred_mape = np.median(concat_mape, axis=0)
MAPE_rmse = np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_mape.flatten())).round(5)

concat_smape = np.concatenate([np.nan_to_num(np.array(smape_pred), nan=0)])
fin_pred_smape = np.median(concat_smape, axis=0)
sMAPE_rmse = np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_smape.flatten())).round(5)

concat_mae = np.concatenate([np.nan_to_num(np.array(mae_pred), nan=0)])
fin_pred_mae = np.median(concat_mae, axis=0)
MAE_rmse = np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_mae.flatten())).round(5)

concat_mse = np.concatenate([np.nan_to_num(np.array(mse_pred), nan=0)])
fin_pred_mse = np.median(concat_mse, axis=0)
MSE_rmse = np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_mse.flatten())).round(5)

performance = np.array([MASE_rmse, MAPE_rmse, sMAPE_rmse, MAE_rmse, MSE_rmse])
os.makedirs('resulttf', exist_ok=True)
pd.DataFrame(performance).to_csv(f"resulttf/trTFTF_{data}_weight.csv")


