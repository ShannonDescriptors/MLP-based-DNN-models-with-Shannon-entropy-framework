### This script could run MLP only model for pchembl/MW prediction (of IC50 of Tissue Factor Pathway Inhibitor) without using Shannon entropy as descriptor and only using the MW as the sole descriptor

# Importing necessary packages
from imutils import paths
import random
import os
import numpy as np
import pandas as pd



# from pyimagesearch.datasets_molpred_2D_1image_resnet import load_house_attributes
# from pyimagesearch.datasets_molpred_2D_1image_resnet import image_data_extract
from KiNet_mlp import KiNet_mlp

from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error


# Getting the list of molecules from csv file obtained from ChEMBL database
df = pd.read_csv('Tissue_Factor_Pathway_Inhibitors_IC50_only_equal_values_with_SMILES.csv', encoding='cp1252') 
# df_target = df['Ligand Efficiency BEI'].values
df_MW = df['Molecular Weight'].values ## The MW for ach molecule
df_target = df['pChEMBL Value'].values
print(df_target)

# Calculating MW normalized target i.e. pchembl/MW
df_target_norm = df['pChEMBL Value'].values ## Defining normalized target
for i in range(0,len(df_target)):
    df_target_norm[i] = df_target[i]/df_MW[i] 
df['pChEMBL Value'] = df_target_norm

print("Shape of the pCHEMBL labels array", df_target.shape)

# # Splitting data into train & tests
print("[INFO] constructing training/ testing split")
split = train_test_split(df, test_size = 0.2, random_state = 42) 

# Distribute the split data
(XtrainTotalData, XtestTotalData) = split  # split format always is in (data_train, data_test, label_train, label_test)
 

#  Grab the maximum/ minimum in the label column
maxPrice = df_target_norm.max()
minPrice = df_target_norm.min() 

# Normalize columns corresponding to max of (pChEMBL Value/MW) values
XtrainLabels  = (XtrainTotalData.iloc[:,-1])/ (maxPrice)
XtestLabels  = (XtestTotalData.iloc[:,-1])/ (maxPrice)

# Defining the 1st column (mol. wt.) as X data
XtrainData = (XtrainTotalData.iloc[:,0:1])
XtestData = (XtestTotalData.iloc[:,0:1])

print("XtrainData shape",XtrainData.shape)
print("XtestData shape",XtestData.shape)


# Max value of the MW column
X_max = max(df.iloc[:,0])

# Normalize the train & test data with X_max i.e. maximum of train & test data
trainContinuous = XtrainTotalData.iloc[:,0:1]/ X_max
testContinuous = XtestTotalData.iloc[:,0:1]/ X_max

# reshape trainContinuous and testContinuous to be able to use them into NN : '.values' is needed to convert them to array from panda dataframe
trainContinuous.values.reshape(-1,1)
testContinuous.values.reshape(-1,1)

print("[INFO] processing input data after normalization....")
XtrainData, XtestData = trainContinuous,testContinuous # train and test arrays as XtrainData and XtestData

# create the model
mlp = KiNet_mlp.create_mlp(1, regress = False) # just one array as input => '1' (1 column feature) 
print("shape of mlp", mlp.output.shape)


# Processing output of MLP model
combinedInput = mlp.output
print("shape of combinedInput (MLP output)",combinedInput.shape)

# Defining the final FC layers 
x = Dense(100, activation = "relu") (combinedInput)
x = Dense(10, activation = "relu") (combinedInput)
x = Dense(1, activation = "linear") (x)
print("shape of x", x.shape)


# Defining final model as 'model' 
model = Model(inputs = mlp.input, outputs = x)
print("shape of mlp input", mlp.input.shape)



# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = SGD(lr= 1.05e-6, decay = 0)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

#train the network
print("[INFO] training network...")

trainY = XtrainLabels
testY = XtestLabels

epoch_number = 500
BS = 5;

# Model fitting across epochs
# Defining the early stop to monitor the validation loss to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')

# shuffle = False to reduce randomness and increase reproducibility
H = model.fit( x = XtrainData, y = trainY, validation_data = (XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 


# evaluate the network
print("[INFO] evaluating network...")
preds = model.predict(XtestData,batch_size=BS)

# compute the difference between the predicted and actual values and then compute the % difference and absolute % difference
diff = preds.flatten() - testY
PercentDiff = (diff/testY)*100
absPercentDiff = (np.abs(PercentDiff)).values

# compute the mean and standard deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )


    
# Plotting Predicted vs Actual values
N = len(testY)
colors = np.random.rand(N)
x = testY * maxPrice
y =  (preds.flatten()) * maxPrice
plt.scatter(x, y, c=colors)
plt.plot( [0,maxPrice],[0,maxPrice] )
plt.xlabel('Actual pchembl/MW', fontsize=18)
plt.ylabel('Predicted pchembl/MW', fontsize=18)
plt.savefig('pchembl_MW_pred_without_shannon_MLP_only.png')
plt.show()


####-------------------------------------------------------------------------------------------------------Evaluating standard statistics of model performance--------------------------------------------------------------------

# ### MAE as a function
# def mae(y_true, predictions):
#     y_true, predictions = np.array(y_true), np.array(predictions)
#     return np.mean(np.abs(y_true - predictions))

# ### The MAE estimated
# print("The mean absolute error estimated: {}".format( mae(x, y) )) 

### The MAPE
print("The mean absolute percentage error: {}".format( mean_absolute_percentage_error(x, y) ) )   

### The MAE
print("The mean absolute error: {}".format( mean_absolute_error(x, y) ))    
    
### The MSE
print("The mean squared error: {}".format( mean_squared_error(x, y) ))  

### The RMSE
print("The root mean squared error: {}".format( mean_squared_error(x, y, squared= False) ) )    

### General stats
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print("The R^2 value between actual and predicted target:", r_value**2)


# plot the training loss and validation loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_number ), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_number ), H.history["val_loss"], label="val_loss")
plt.title("Training loss and Validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss@epochs_pchembl_MW_without_shannon_MLP_only')    
####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------