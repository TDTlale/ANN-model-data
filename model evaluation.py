# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 19:37:25 2025

@author: tlale
"""
%reset -f

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import loadtxt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import csv
from matplotlib.ticker import FormatStrFormatter
from metrics import ape
from metrics import mse
from metrics import R2_score
from cfunctions import label_loc
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib as mpl

#get directory
cd = os.path.dirname(os.getcwd())

#process variable names
columns = ['Laser Power', 'Scan Speed', 'Powder Feed Rate', 'Height', 'Width', 'Dilution' ]

#load trained artificial neural network model
ANN_model = load_model('ANN_model_pso.keras')

#import experimental data sheets
unscaled_data = pd.ExcelFile(r"{}\unscaled_data.xlsx".format(cd))

#import training data
xtrain_unscaled = pd.read_excel(unscaled_data, 'xtrain').iloc[:,:].values
ytrain_unscaled = pd.read_excel(unscaled_data, 'ytrain').iloc[:,:].values

#import validation data
xvalid_unscaled  = pd.read_excel(unscaled_data, 'xvalid').iloc[:,:].values
yvalid_unscaled  = pd.read_excel(unscaled_data, 'yvalid').iloc[:,:].values

#import test data
xtest_unscaled = pd.read_excel(unscaled_data, 'xtest').iloc[:,:].values
ytest_unscaled = pd.read_excel(unscaled_data, 'ytest').iloc[:,:].values

#initializing scaling function
sc = MinMaxScaler()

#fit scaling function for inputs
sc.fit(xtrain_unscaled)

#transform training and testing inputs
xtrain_scaled = sc.transform(xtrain_unscaled)
xtest_scaled = sc.transform(xtest_unscaled)

#make predictions of training and testing data
train_predictions = ANN_model.predict(xtrain_scaled)
test_predictions = ANN_model.predict(xtest_scaled)

#fit scaling function for outputs
sc.fit(ytrain_unscaled)

#tranform expiremental training and testing ouputs
ytrain_scaled = sc.transform(ytrain_unscaled)
ytest_scaled = sc.transform(ytest_unscaled)

#transform training and testing predictions
y_pred_train = sc.inverse_transform(train_predictions)
y_pred_test = sc.inverse_transform(test_predictions)


# training metrics
print('\n ___Training Metrics___')
print(f'{ANN_model.evaluate(xtrain_scaled,ytrain_scaled, verbose = 0)}') 
model_evaluation = R2_score(ytrain_scaled, train_predictions)
train_ape = ape(ytrain_unscaled,y_pred_train)

# testing metrics 
print(f'\n ___Testing Metrics___')
print(f'{ANN_model.evaluate(xtest_scaled,ytest_scaled, verbose = 0)} \n')
print(np.round(ape(ytest_unscaled,y_pred_test),2))
test_mse = mse(ytest_scaled, test_predictions)

axis_font = {'family': 'Palatino Linotype','size': 18}
title_font = {'family': 'Palatino Linotype','size': 20}
tick_font = {'family': 'Palatino Linotype','size': 16}

labels = ['Height', 'Width', 'Dilution']
units = {'Height': '(mm)','Width': '(mm)',  'Dilution': ''}

fig , axs = plt.subplots(1,3,figsize=(15, 5), sharey = None, layout="tight")
fig.subplots_adjust(hspace = 0.2)
 
labels = ['Height', 'Width', 'Dilution']
units = {'Height': '(mm)','Width': '(mm)',  'Dilution': ''}

for sample in range(0,len(xtrain_unscaled)):
    
    for index in range(ytrain_unscaled.shape[1]):
        feature =labels[index]
        
        axs[index].scatter(ytrain_unscaled[sample][index], y_pred_train[sample][index], marker= 'o', facecolors='none', edgecolors='r')
        axs[index].set_title(feature,font = title_font)
        axs[index].annotate(f'$R^2$ = {round(model_evaluation[index],3)}', xy=(0.1,0.9),xycoords='axes fraction', font = tick_font)
        axs[index].set_ylabel(f'Prediction {units[feature]}',font = axis_font)
        axs[index].set_xlabel(f'Target {units[feature]}', font = axis_font)

for ax in axs[:4]:
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_font(tick_font)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_font(tick_font)

axs[0].plot(ytrain_unscaled[:,0], ytrain_unscaled[:,0], linestyle = 'solid', color = 'k')
axs[1].plot(ytrain_unscaled[:,1], ytrain_unscaled[:,1], linestyle = 'solid', color = 'k')
axs[2].plot(ytrain_unscaled[:,2], ytrain_unscaled[:,2], linestyle = 'solid', color = 'k')

plt.show()
