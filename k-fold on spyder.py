%reset

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import initializers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#import unscaled raw-data
unscaled_data = pd.ExcelFile(r"{}\unscaled_data.xlsx".format(os.path.dirname(os.getcwd())))

xtrain = pd.read_excel(unscaled_data, 'xtrain').iloc[:,:].drop([3], axis = 1).values
ytrain = pd.read_excel(unscaled_data, 'ytrain').iloc[:,:].values

xvalid  = pd.read_excel(unscaled_data, 'xvalid').iloc[:,:].drop([3], axis = 1).values
yvalid  = pd.read_excel(unscaled_data, 'yvalid').iloc[:,:].values


#concatenate training and validation data into one
x = np.concatenate((xtrain,xvalid), axis = 0)
y = np.concatenate((ytrain,yvalid), axis = 0)


#K-fold
split = 4
kfold = KFold(n_splits = split, shuffle= True)

acc_per_fold = []
loss_per_fold = []

EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',  
                                            mode='min',
                                            min_delta = 1e-7,
                                            verbose = 0, 
                                            start_from_epoch = 0)

def create_model():
    

    Optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,
                                    beta_1 = 0.64,
                                    beta_2 = 0.67
                                    )

    

    #delete existing model and cache
    try:
        del model
    except:
        print('Model Cleared')
        
    #create instance of model and define architecture    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units = 8,
                                  kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 1),
                                  activation = 'tanh',
                                  use_bias=True,
                                  bias_initializer="zeros"
                                  ),
            
            tf.keras.layers.Dense(units = 3,
                                  kernel_initializer = tf.keras.initializers.HeUniform(seed = 1),
                                  activation = 'linear',
                                  use_bias=True,
                                  bias_initializer="zeros"
                                  )
        ]
    )
    
    #compile model
    model.compile(optimizer =  Optimizer,
                   loss = 'mse', 
                   metrics = ['r2_score'])
    return model


plots = {}
    
fold_no = 1

for train, validate in kfold.split(x, y):
    
    
    #scaling data
    sc = MinMaxScaler()
    x_train = sc.fit_transform(x[train])
    x_val = sc.transform(x[validate])
    
    y_train = sc.fit_transform(y[train])
    y_val = sc.transform(y[validate])
   # y_test = sc.transform(ytest)
    
        

    ANN = create_model()
    #Train the model
    training = ANN.fit(x_train, y_train,
                         epochs = 10000,
                         batch_size = 15,
                         validation_data = (x_val, y_val),
                         callbacks = EarlyStop,
                         verbose = 0)
    
    #Archive training histories in dictionary
    plots[f'FOLD {fold_no}'] = training.history
    
    #Evalute Model Performance on Test Data
    print(f'FOLD {fold_no} PERFORMANCE:')
    scores = ANN.evaluate(x_val, y_val, verbose = 1)
    
    #Archive Test Metrics in list
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    
    fold_no = fold_no + 1
    

#PLOTS-----------------------------------------------------------------------------------------------------------------
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Fold')
ax1.set_ylabel('Accuracy: R\u00b2', color=color)
ax1.plot(range(1,len(acc_per_fold)+1,1), acc_per_fold, color=color, marker = '.')
ax1.set_ylim((0,1))
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Loss: MSE', color=color)  # we already handled the x-label with ax1
ax2.plot(range(1,len(acc_per_fold)+1,1), loss_per_fold, color=color, marker = '.')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout
ax1.set_title('K-Fold Cross-Validation Metrics')
plt.show()


start = 0
taining_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

for key in plots.keys():
    
    end = start + len(plots[key]['loss'])
    
    taining_loss.append(plots[key]['loss'])
    validation_loss.append(plots[key]['val_loss'])
    
    training_accuracy.append(plots[key]['r2_score'])
    validation_accuracy.append(plots[key]['val_r2_score'])
    
    start  = end 

start = 0
plt.plot(list(range(start,end)),sum(taining_loss,[]), linestyle = 'solid', color = 'k', linewidth = 1, label = 'Training')
plt.plot(list(range(start,end)),sum(validation_loss,[]), linestyle = 'dashed', color = 'r', linewidth = 1, label = 'Validation')
plt.title('K-Fold Cross-Validation')
plt.ylabel('Loss: MSE')
plt.xlabel('Epochs')
xmin, xmax, ymin, ymax = plt.axis()
plt.xlim((0,xmax+100))
plt.legend( loc = 'best')
plt.show()

plt.plot(list(range(start,end)),sum(training_accuracy,[]), linestyle = 'solid', color = 'k', linewidth = 1, label = 'Training')
plt.plot(list(range(start,end)),sum(validation_accuracy,[]), linestyle = 'dashed', color = 'r', linewidth = 1, label = 'Validation')
plt.title('K-Fold Cross-Validation')
plt.ylabel('Accuracy: R\u00b2')
plt.xlabel('Epochs')
xmin, xmax, ymin, ymax = plt.axis()
plt.xlim((0,xmax+100))
plt.legend( loc = 'best')
plt.show()
