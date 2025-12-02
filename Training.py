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

#importing data
unscaled_data = pd.ExcelFile(r"{}\unscaled_data.xlsx".format(os.path.dirname(os.getcwd())))

xtrain = pd.read_excel(unscaled_data, 'xtrain').iloc[:,:].drop([3], axis = 1).values
ytrain = pd.read_excel(unscaled_data, 'ytrain').iloc[:,:].values

#drop([3], axis = 1)

xvalid  = pd.read_excel(unscaled_data, 'xvalid').iloc[:,:].drop([3], axis = 1).values
yvalid  = pd.read_excel(unscaled_data, 'yvalid').iloc[:,:].values

xtest = pd.read_excel(unscaled_data, 'xtest').iloc[:,:].drop([3], axis = 1).values
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
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',  
                                            mode='min',
                                            verbose = 1, 
                                            min_delta = 1e-7, 
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
plt.ylabel('R2 Score')
plt.xlabel('Epochs')
plt.legend(['Training','Validating'], loc = 'best')
plt.show()

plt.plot(history.history['loss'], color = 'k')
plt.plot(history.history['val_loss'], linestyle = 'dashed', color = 'r')
plt.title('Learning Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training','Validating'], loc = 'best')
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




for feature in range(ytrain.shape[1]):
    
    units = {'Height': '(mm)','Width': '(mm)',  'Dilution': ''}
    labels = ['Height', 'Width', 'Dilution']
    
    prop = labels[feature]
    
    plt.scatter( np.arange(1, len(ytrain)+1), ytrain[:, feature], marker = 'o', color = 'b', label = 'Experiment')
    plt.plot( np.arange(1, len(y_pred_train)+1), y_pred_train[:, feature], color = 'r', label = 'Prediction')
    plt.xlabel('Sample', fontsize=14)
    plt.ylabel(f'{prop} {units[prop]}', fontsize=14)
    plt.legend(fontsize = 8)
    plt.show()



for feature in range(ytrain.shape[1]):
    
    labels = ['Height', 'Width', 'Dilution']
    units = {'Height': '(mm)','Width': '(mm)',  'Dilution': ''}
    fig , axs = plt.subplots(2,1,figsize=(10, 5), sharey = None)
    #fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0)
    fig.align_labels()

    
    for sample in range(0,len(xtrain)):
        Feature =labels[feature]
        axs[0].scatter(sample+1,ytrain[sample][feature], marker= 's', facecolors='none', edgecolors='k')
        axs[0].scatter(sample+1,y_pred_train[sample][feature], marker = 'o', facecolors='none', edgecolors='r')
        axs[0].set_ylabel(f'{Feature} {units[Feature]}')
        axs[0].set_xlabel('Sample' )
        
        axs[1].bar(sample+1, round(train_ape[sample][feature],2), color = 'white',edgecolor='black', hatch="//", width=0.5) #f'{round(train_ape[sample][0],2)}%')
        axs[1].set_ylabel('APE (%)')
        axs[1].grid(visible = True, which='major', axis='y')
    plt.show()


fig , axs = plt.subplots(1,3,figsize=(15, 5), sharey = None, layout="constrained")
fig.subplots_adjust(hspace = 0.2)

for sample in range(0,len(xtrain)):
    
    axs[ 0].scatter(ytrain[sample][0], y_pred_train[sample][0], marker= 'o', facecolors='none', edgecolors='r')
    axs[0].set_title('Height',fontsize=20)
    axs[0].annotate(f'$R^2$ = {round(model_evaluation[0],3)}', xy=(0.1,0.9),xycoords='axes fraction', fontsize=14)
    axs[0].set_ylabel('Prediction (mm)',fontsize=14)
    axs[0].set_xlabel('Target (mm)',fontsize=14)

    
    axs[ 1].scatter(ytrain[sample][1], y_pred_train[sample][1], marker= 'o', facecolors='none', edgecolors='r')
    axs[1].set_title(f'Width',fontsize=20)
    axs[1].annotate(f'$R^2$ = {round(model_evaluation[1],3)}', xy=(0.1,0.9),xycoords='axes fraction', fontsize=14)
    axs[1].set_ylabel('Prediction (mm)',fontsize=14)
    axs[1].set_xlabel('Target (mm)',fontsize=14)

    
    axs[ 2].scatter(ytrain[sample][2], y_pred_train[sample][2], marker= 'o', facecolors='none', edgecolors='r')
    axs[2].set_title(f'Dilution',fontsize=20)
    axs[2].annotate(f'$R^2$ = {round(model_evaluation[2],3)}', xy=(0.1,0.9),xycoords='axes fraction', fontsize=14)
    axs[2].set_ylabel('Prediction',fontsize=14)
    axs[2].set_xlabel('Target',fontsize=14)
    
    #axs[1 , 1].scatter(ytrain[sample][3], y_pred_train[sample][3], marker= 'o', facecolors='none', edgecolors='b')
   # axs[1 , 1].set_title(f'Dilution \n [$R^2$ = {round(model_evaluation[3],3)}, MSE = {round(MSE[3],3)}]')


axs[0].plot(ytrain[:,0], ytrain[:,0], linestyle = 'solid', color = 'k')
axs[1].plot(ytrain[:,1], ytrain[:,1], linestyle = 'solid', color = 'k')
axs[2].plot(ytrain[:,2], ytrain[:,2], linestyle = 'solid', color = 'k')
#axs[0 , 1].plot(ytrain[:,3], ytrain[:,3], linestyle = 'solid', color = 'k')
plt.show()

#ANN.save('ANN_model_pso.keras')

tf.keras.backend.clear_session()


sampling =[]

for i in range(0,100):
    import random
    
    numbers = range(1, 15)
    random_sample = random.sample(numbers, 4)
    
    #print(x_train[random_sample])
    
    sampled_predictions = ANN.predict(x_train[random_sample], verbose = 0)
    
    sampled_evaluation = mse(y_train[random_sample],sampled_predictions)
    
    sampling.append(sum(sampled_evaluation)/3)
    
    
fig = plt.figure(figsize =(10, 7))

plt.boxplot(sampling, orientation='horizontal')
plt.scatter(sum(test_mse/3), 1, color = 'red')
plt.show()