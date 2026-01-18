# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 08:07:41 2024

@author: Dell
"""
%reset -f

#importing libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from metrics import ape
from metrics import mse
from metrics import R2_score
import matplotlib.pyplot as plt
from keras import initializers
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#get directory
cd = os.path.dirname(os.getcwd())

#importing data
unscaled_data = pd.ExcelFile(r"{}\unscaled_data.xlsx".format(os.path.dirname(os.getcwd())))

xtrain = pd.read_excel(unscaled_data, 'xtrain').iloc[:,:].values
ytrain = pd.read_excel(unscaled_data, 'ytrain').iloc[:,:].values

#drop([3], axis = 1)

xvalid  = pd.read_excel(unscaled_data, 'xvalid').iloc[:,:].values
yvalid  = pd.read_excel(unscaled_data, 'yvalid').iloc[:,:].values

xtest = pd.read_excel(unscaled_data, 'xtest').iloc[:,:].values
ytest = pd.read_excel(unscaled_data, 'ytest').iloc[:,:].values

# normalizing data
sc = MinMaxScaler()

x_train = sc.fit_transform(xtrain)
x_valid = sc.transform(xvalid)
x_test = sc.transform(xtest)

y_train = sc.fit_transform(ytrain)
y_valid = sc.transform(yvalid)
y_test = sc.transform(ytest)

# define optimizer
Optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,
                                beta_1 = 0.64,
                                beta_2 = 0.67
                                )
# define early stopping
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
                                            mode='min',
                                            verbose = 1, 
                                            min_delta = 1e-10,
                                            patience = 10,
                                            restore_best_weights = True,
                                            start_from_epoch = 0)
# create model
def create_model():
    
    # delete existing model and cache
    try:
        del model
    except:
        print('Model Cleared')
        
    # create instance of model and define architecture    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units = 8,
                                  kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 1),
                                  use_bias = True,
                                  bias_initializer='zeros',
                                  activation = 'tanh'
                                  ),
            
            tf.keras.layers.Dense(units = 3,
                                  kernel_initializer = tf.keras.initializers.HeUniform(seed = 1),
                                  use_bias = True,
                                  bias_initializer='zeros',
                                  activation = 'linear'
                                  )
        ]
    )
    
    # compile model
    model.compile(optimizer =  Optimizer,
                   loss = 'mse', 
                   metrics = ['r2_score'])
    return model

# create model
ANN = create_model()

# train model
history = ANN.fit(x_train, y_train,
                     epochs = 3000,
                     validation_data = (x_valid,y_valid),
                     callbacks = EarlyStop,
                     verbose = 0)


# Plotting Performace Metrics
plt.plot(history.history['r2_score'], color = 'k')
plt.plot(history.history['val_r2_score'], linestyle = 'dashed', color = 'r')
plt.title('Learning Accuracy')
plt.ylabel('Accuracy: R\u00b2')
plt.xlabel('Epochs')
plt.legend(['Training','Validating'], loc = 'best')
plt.savefig(f"{cd}/plots/training_r2_score.png", dpi = 300, bbox_inches ='tight')
plt.show()

plt.plot(history.history['loss'], color = 'k')
plt.plot(history.history['val_loss'], linestyle = 'dashed', color = 'r')
plt.title('Learning Loss')
plt.ylabel('Loss: MSE')
plt.xlabel('Epochs')
plt.legend(['Training','Validating'], loc = 'best')
plt.savefig(f"{cd}/plots/training_MSE.png", dpi = 300, bbox_inches ='tight')
plt.show()


# make predictions on training and testing data
train_predictions = ANN.predict(x_train)
test_predictions = ANN.predict(x_test)

# scale the predictions to normal scale
sc.fit(ytrain)
y_pred_train = sc.inverse_transform(train_predictions)
y_pred_test = sc.inverse_transform(test_predictions)

# training metrics
print('\n ___Training Metrics___')
print(f'{ANN.evaluate(x_train,y_train, verbose = 0)}') 
model_evaluation = R2_score(y_train, train_predictions)
train_ape = ape(ytrain,y_pred_train)

# testing metrics 
print(f'\n ___Testing Metrics___')
print(f'{ANN.evaluate(x_test,y_test, verbose = 0)} \n')
print(np.round(ape(ytest,y_pred_test),2))
test_mse = mse(y_test, test_predictions)


#save training history to an excel file
df = pd.DataFrame(history.history)

# Save to Excel
df.to_excel(f"{cd}/raw_data/model_training_history.xlsx", index=False)


ANN.save('ANN_model_pso.keras')

