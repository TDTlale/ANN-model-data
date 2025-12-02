# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:07:34 2024

@author: Dell
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
xtrain = pd.read_excel(unscaled_data, 'xtrain').iloc[:,:].values
ytrain = pd.read_excel(unscaled_data, 'ytrain').iloc[:,:].values

#import validation data
xvalid  = pd.read_excel(unscaled_data, 'xvalid').iloc[:,:].values
yvalid  = pd.read_excel(unscaled_data, 'yvalid').iloc[:,:].values

#import test data
xtest = pd.read_excel(unscaled_data, 'xtest').iloc[:,:].values
ytest = pd.read_excel(unscaled_data, 'ytest').iloc[:,:].values

#entire experimental data
x_exp = np.concatenate((xtrain,xvalid,xtest), axis = 0)
y_exp = np.concatenate((ytrain,yvalid,ytest), axis = 0)
exp_data = pd.DataFrame(np.concatenate((x_exp,y_exp), axis = 1), columns = columns)



# import prediction inputs
x_pred = pd.read_excel(pd.ExcelFile(r"{}\prediction_array.xlsx".format(cd)), 'prediction_array').iloc[:,:].values

#initializing scaling function
sc = MinMaxScaler()
#fit scaling function 
sc.fit(xtrain)
#make predictions based on array
pred_outputs = ANN_model.predict(sc.transform(x_pred))

test_inputs = sc.transform(xtest)

#opt_outputs = ANN_model.predict(sc.transform([[1400, 500,6.36]]))
y_test = ANN_model.predict(sc.transform(xtest))
y_train = ANN_model.predict(sc.transform(xtrain))
#scale prediction outputs



sc.fit(ytrain)
test_outputs = sc.transform(ytest)
y_pred = sc.inverse_transform(pred_outputs)
#opt_sample = sc.inverse_transform(opt_outputs)
test_pred = sc.inverse_transform(y_test)
train_pred = sc.inverse_transform(y_train)

print(f'{ANN_model.evaluate(test_inputs, test_outputs, verbose = 0)} \n')
error = ape(ytest,test_pred)
train_error = ape(ytrain, train_pred)

# concatenate prediction inputs and outputs into dataframe
pred_data = pd.DataFrame(np.concatenate((x_pred,y_pred), axis = 1), columns = columns)


'''
Process Maps
'''
pairs = [('Laser Power', 'Scan Speed'), ('Laser Power', 'Powder Feed Rate'),('Scan Speed', 'Powder Feed Rate')]
output_units = {'Height': '(mm)','Width': '(mm)',  'Dilution': '', 'Aspect Ratio': ''}
input_units = {'Laser Power': 'W','Scan Speed': 'mm/min',  'Powder Feed Rate': 'g/min'}
input_symbols= {'Laser Power': 'P','Scan Speed': 'S',  'Powder Feed Rate': 'F'}
input_props = columns[:3]
output_props  = columns[3:]
output_props.append('Aspect Ratio')
input_filters = {'Laser Power': 1500, 'Scan Speed': 500, 'Powder Feed Rate':7.0}


for Property in ['Dilution', 'Width', 'Height']:
    index =  output_props.index(Property)
    pairs_dict = {}

    fig0 = plt.figure(figsize=plt.figaspect(0.45), layout = 'constrained')

    for pair in pairs:
        
        pair_index =  pairs.index(pair)
        a, b = pair
        
        diff = list(set(input_props) - set([a, b]))[0]

        filter_value = input_filters[diff]
        
        #print(f'{Property} vs {a} and {b}, {diff} @ {filter_value}')
        
        power_pred = pred_data.loc[  (pred_data[diff] == filter_value)]
        
        X = np.array(power_pred[b])
        Y = np.array(power_pred[a])


        x_set = list(set(X))
        y_set = list(set(Y))

        x_set.sort()
        y_set.sort()

        x_params = np.array(x_set)
        y_params = np.array(y_set)
        
        
        Z = []
        
        for LP in list(y_params):
            row = []
        
            for SS in list(x_params):
        
                z =  power_pred.loc[(np.array(power_pred[a])== LP) & (np.array(power_pred[b]) == SS)]
                
                prop  = float(z[Property].values)
                row.append(round(prop,3))
            Z.append(row)
        
        Z = np.array(Z)
        
        x_label = f'{input_symbols[b]} ({input_units[b]})'
        y_label = f'{input_symbols[a]} ({input_units[a]})'
        z_label = f'{Property} {output_units[Property]}'
        
        axis = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 6,
        }
        
        
        labels = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 8,
        }
        
        x, y = np.meshgrid(x_params, y_params)

        # get min and max values
        xmin,xmax = (np.min(x),np.max(x))
        ymin,ymax = (np.min(y),np.max(y))
        zmin,zmax = (np.min(Z),np.max(Z))
        
        # create 3d axes
        ax = fig0.add_subplot(1, 3, pair_index +1, projection='3d')
        ax.grid(visible=None) #diable grid lines
        # add surface plot
        ax.plot_surface(x, y, Z,edgecolors = 'k',cmap = 'rainbow', linewidth = 0, antialiased=False)
        # add wireframe plot
        ax.plot_wireframe(x, y, Z,linewidth = 0.5, edgecolors='k')
        # add contour plot to x-y plane
        cs = ax.contourf(x, y, Z, 5,zdir = 'z', offset = 0.95*min(pred_data[Property]), cmap = 'rainbow')
        
        # create tittle
        ax.text2D(0.25, 0.90, f'{input_symbols[diff]} fixed at {input_filters[diff]} {input_units[diff]}', transform=ax.transAxes)
        # axis properties
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params( labelsize = 6)                                  # set tick labels size 
        ax.tick_params(axis ='y', labelsize = 6, labelrotation = -10)   # set tick labels size and orientation for z-axis
        ax.set_xlabel(x_label,labelpad = 1.5, fontdict = labels)        # create label for x-axis
        ax.set_ylabel(y_label,labelpad = 1.5, fontdict = labels)        # create label for y-axis
        ax.set_zlabel(z_label,labelpad = 1.5, fontdict = labels)        # create label for z-axis
        
        cbar = fig0.colorbar(cs,location = 'bottom', orientation = 'horizontal', fraction = 0.035, pad = 0.05)
        cbar.ax.set_title(z_label, fontdict = labels)
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar.ax.tick_params( labelsize = 6)
        
        ax.view_init(15,-65)
        
        ax.set_zlim([0.95*min(pred_data[Property]),
                     1.05*max(pred_data[Property])])
        ax.set_box_aspect(None, zoom = 0.85)

    plt.show()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
pred_data.insert(3, 'Powder Density', np.array(pred_data['Powder Feed Rate']/pred_data['Scan Speed']))
pred_data.insert(4, 'Energy Density', np.array((pred_data['Laser Power']*60)/(2*pred_data['Scan Speed'])))

pred_data.replace([np.inf, -np.inf], np.nan, inplace=True)
pred_data.dropna(subset=['Powder Density'], how="all", inplace=True)
pred_data.insert(8, 'Aspect Ratio', np.array((pred_data['Width'])/(pred_data['Height'])))





line_style = {'Dilution': 'solid', 'Width':'dotted', 'Height':'dashed', 'Aspect Ratio':'solid'}
line_color = {'Dilution': 'k', 'Width':'k', 'Height':'k', 'Aspect Ratio':'k'}
prop_symbol = {'Dilution': 'D', 'Width':'w', 'Height':'h', 'Aspect Ratio':'AR'}

level = {'Dilution':[0.1, 0.14, 0.24, 0.3, 0.4,0.5,0.6], 
         'Width': [2.6,2.8,3,3.2,3.4], 
         'Height':[0.2,0.4,0.6,0.8,1,1.2], 
         'Aspect Ratio':[2.5]}


fig, axs = plt.subplots(figsize=(8,5), sharey = None, layout="constrained")
figs = plt.figure(figsize=plt.figaspect(0.35), layout="constrained")


D = {}
AR = {}

for Property in ['Height', 'Width', 'Aspect Ratio', 'Dilution']:
    # get index of property
    index =  output_props.index(Property)
    # create dataframe for scan speed fixed at 500 mm/min
    power_pred = pred_data.loc[(pred_data['Scan Speed'] == 500)]
    '''if Property == 'Aspect Ratio':
        power_pred = power_pred.drop(power_pred[(power_pred['Powder Feed Rate']<4)].index)'''
    # create X and Y variables    
    X = np.array(power_pred['Powder Feed Rate'])
    Y = np.array(power_pred['Laser Power'])
    
    # create list of unique variables for x and y 
    x_set = list(set(X))
    y_set = list(set(Y))
    
    # sorting the variables
    x_set.sort()
    y_set.sort()

    # convert the lists into numpy arrays
    x_params = np.array(x_set)
    y_params = np.array(y_set)
    
    # create empty list fot depended variables
    Z = []
   
    for LP in list(y_params):
        row = []
        
        for SS in list(x_params):
            
            z =  power_pred.loc[(np.array(power_pred['Laser Power']) == LP) & 
                                (np.array(power_pred['Powder Feed Rate']) == SS)]
            
            prop  = float(z[Property].values)
            row.append(round(prop,4))
        
        Z.append(row)
    
    #Z = pred_data[Property]
    
    Z = np.array(Z)
        
    x_label = 'Powder Feed Rate (g/min)'
    y_label = 'Laser Power (W)'
    
    
    x, y = np.meshgrid(x_params, y_params)
       
       
    
    #create line contour plot
    x, y = np.meshgrid(x_params, y_params)
   
    class nf(float):
        def __repr__(self):
            s = f'{self:.2f}'
            return f'{self:.1f}' if s[-1] == '0' else s
    
    
        
    CS = axs.contour(x, y, Z, levels = level[Property], colors = line_color[Property], linestyles = line_style[Property], linewidths=0.5)
    
    
    if Property == 'Dilution':
        mpl.rcParams['hatch.linewidth'] = 0.3
        axs.contourf(x, y, Z,levels = [0.14,0.24], colors = 'none', hatches='\\' )


    elif Property == 'Aspect Ratio':

        axs.contourf(x, y, Z,levels = [0.0,2.5],colors = 'red', alpha=0.1)
 
    
    if Property == 'Dilution':
        
        level, loc, indices = label_loc(CS, 0.95, 0.94, prop_symbol, Property)
        CS.clabel(manual = loc, fontsize = 9, fmt = level)    
    else:
        # Recast levels to new class
        CS.levels = [nf(val) for val in CS.levels]
        
        # Label levels with specially formatted floats
        if plt.rcParams["text.usetex"]:
            fmt = r'{prop_symbol[Property]} = %r%'
        else:
            fmt = f'{prop_symbol[Property]} = %r'
        
        axs.clabel(CS, inline =True, fontsize = 9, fmt = fmt)
                     


    axs.set_xlabel(x_label, fontsize = 10)
    axs.set_ylabel(y_label, fontsize = 10)
    axs.set_title('Process Map at Fixed Scan Speed of 500 mm/min', fontsize = 10)
    

lines =[]
for key in line_style.keys():
    lines.append(Line2D([0], [0], color=line_color[key], lw=1, linestyle = line_style[key] , label= f'{key} ({prop_symbol[key]})'))

axs.legend(handles=lines, bbox_to_anchor=(1.05, 1),loc = 'upper left', title ='Note: h & w in mm', alignment='left')
t1 = ("Processing Window")
t2 = ("Risk Lack of Fusion")
axs.text(5.3, 1200, t1, ha='left', rotation=60, wrap=True)
axs.text(6, 1050, t2, ha='left', rotation=60, wrap=True)  
axs.scatter(6.41,1394, label = 'TMMC10', color = 'red')
axs.annotate('Test Sample', (6.1,1400), fontsize = 9)
plt.show()   


