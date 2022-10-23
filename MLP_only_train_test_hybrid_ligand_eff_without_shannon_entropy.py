### This script could run an MLP only model for predicting BEI values of molecules (of IC50 of Tissue Factor Pathway Inhibitor) without using Shannon entropy as descriptor and only using the MW as the sole descriptor



# import the necessary packages
import numpy as np
import pandas as pd


from KiNet_mlp import KiNet_mlp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error



# Getting the list of molecules from csv file obtained from ChEMBL database
df = pd.read_csv('Tissue_Factor_Pathway_Inhibitors_IC50_only_equal_values_ligand_eff.csv', encoding='cp1252') 
df_target = df['Ligand Efficiency BEI/MW'].values
print(df_target)
print("Shape of the pCHEMBL labels array", df_target.shape)


print("[INFO] constructing training/ testing split")
split = train_test_split(df, test_size = 0.2, random_state = 42) 

# Distribute the input data columns in train & test splits
(XtrainTotalData, XtestTotalData) = split  # split format always is in (data_train, data_test, label_train, label_test)


# Taking only the (Ligand Efficiency BEI/MW) as the target: Getting the extreme values of target column
maxPrice = df.iloc[:,-1].max() 
minPrice = df.iloc[:,-1].min() 
print(maxPrice,minPrice)
XtrainLabels  = (XtrainTotalData.iloc[:,-1])/ (maxPrice)
XtestLabels  = (XtestTotalData.iloc[:,-1])/ (maxPrice)

# Taking just the 1st column (mol. wt.) as X data
XtrainData = (XtrainTotalData.iloc[:,0])
XtestData = (XtestTotalData.iloc[:,0])
print("XtrainData shape",XtrainData.shape)
print("XtestData shape",XtestData.shape)


# perform min-max scaling each continuous (only 1) feature column to the range [0 1]
X_max = max(df.iloc[:,0])
trainContinuous = XtrainTotalData.iloc[:,0]/ X_max
testContinuous = XtestTotalData.iloc[:,0]/ X_max

# reshape trainContinuous and testContinuous to be able to use them into NN : '.values' is needed to convert them to array from panda dataframe
trainContinuous.values.reshape(-1,1)
testContinuous.values.reshape(-1,1)

print("[INFO] processing input data after normalization....")
XtrainData, XtestData = trainContinuous,testContinuous ## Feeding single array as XtrainData and XtestData

# create the MLP model
mlp = KiNet_mlp.create_mlp(1, regress = False) # just one array as input => '1' (1 column feature) 
print("shape of mlp", mlp.output.shape)

# Processing the MLP output
combinedInput = mlp.output
print("shape of combinedInput",combinedInput.shape)

# Defining the final FC (Dense) layers 
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
H = model.fit( x = XtrainData , y = trainY, validation_data = ( XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 


# evaluate the network
print("[INFO] evaluating network...")
preds = model.predict(XtestData,batch_size=BS)

# compute the difference between the predicted and actual house prices, then compute the % difference and actual % difference
diff = preds.flatten() - testY
PercentDiff = (diff/testY)*100
absPercentDiff = (np.abs(PercentDiff)).values

# compute the mean and standard deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )

    
## Plotting Predicted  vs Actual Ligand BEI values
N = len(testY)
colors = np.random.rand(N)
x = testY * maxPrice
y =  (preds.flatten()) * maxPrice
plt.scatter(x, y, c=colors)
plt.plot( [0,maxPrice],[0,maxPrice] )
plt.xlabel('Actual BEI', fontsize=18)
plt.ylabel('Predicted BEI', fontsize=18)
plt.savefig('Ligand_Efficiency_BEI_pred_without_shannon_MLP_only.png')
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
plt.savefig('loss@epochs_BEI_without_shannon_MLP_only')    
####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
