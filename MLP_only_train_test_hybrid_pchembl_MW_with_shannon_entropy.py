### This script could run MLP only model for pchembl/MW prediction (of IC50 of Tissue Factor Pathway Inhibitor) using Shannon entropy as descriptor along with the MW as the other descriptor


# import the necessary packages
from imutils import paths
import random
import os
import numpy as np
import pandas as pd


from KiNet_mlp import KiNet_mlp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error

from rdkit import Chem

import re
import math

# Getting the list of molecules from csv file obtained from ChEMBL database
df = pd.read_csv('Tissue_Factor_Pathway_Inhibitors_IC50_only_equal_values_with_SMILES.csv', encoding='cp1252') 
df_MW = df['Molecular Weight'].values ## The MW for ach molecule
df_target = df['pChEMBL Value'].values
# print(df_target)

# Calculating MW normalized target i.e. pchembl/MW
df_target_norm = df['pChEMBL Value'].values ## Defining normalized df_target
for i in range(0,len(df_target)):
    df_target_norm[i] = df_target[i]/df_MW[i] 
df['pChEMBL Value'] = df_target_norm
print("Shape of the pChEMBL Value/MW labels array", df_target.shape)    


#  Grab the maximum/ minimum in the label column
maxPrice = df_target_norm.max() # grab the maximum price in the training set's label column
minPrice = df_target_norm.min() # grab the minimum price in the training set's label column


###------------------------------------------------------Shannon Entropy Generation: SMILES/ SMARTS/ InChiKey--------------------------------------------------------------------------------------------------

### comment this below section if SMILES Shannon is not used
###---------------------------------------------------------------------SMILES Shannon-------------------------------------------------------------------------------------------------------------------------
# Generate a new column with title 'shannon_smiles'. Evaluate the Shannon entropy for each smile string and store into 'shannon_smiles' column of df table

### Inserting the new column as the 2nd column in df
df.insert(1, "shannon_smiles", 0.0)

# smiles regex definition
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)

shannon_arr = []
for p in range(0,len(df['shannon_smiles'])):
    
    molecule = df['Smiles'][p]
    tokens = regex.findall(molecule)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)

    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)
        
    print("shannon entropy: ", shannon)
    shannon_arr.append(shannon)
        
    
df['shannon_smiles']= shannon_arr

###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### comment this below section if SMARTS Shannon is not used
###---------------------------------------------------------------------SMARTS Shannon-------------------------------------------------------------------------------------------------------------------------
### Generate a new column with title 'shannon_smarts'. Evaluate the Shannon entropy for each smile string and store into 'shannon_smarts' column of df table

### Inserting the new column as the 2nd column in df
df.insert(1, "shannon_smarts", 0.0)


### smarts regex definition
smarts_REGEX_PATTERN =  r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(smarts_REGEX_PATTERN)


shannon_arr = []
for p in range(0,len(df['shannon_smarts'])):
    
    smiles = df['Smiles'][p]
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)
    m_smarts = Chem.MolToSmarts(mol)
    
    molecule = m_smarts
    tokens = regex.findall(molecule)
    # print(tokens)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)
    
        
    print("shannon entropy on smarts: ", shannon)
    shannon_arr.append(shannon)  
    
df['shannon_smarts']= shannon_arr  

###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### comment this below section if InChiKey Shannon is not used
###---------------------------------------------------------------------InChiKey Shannon-------------------------------------------------------------------------------------------------------------------------
### Generate a new column with title 'shannon_inchikey'. Evaluate the Shannon entropy for each smile string and store into 'shannon_inchikey' column of df table


### InChiKey: reading all letters except special characters
df.insert(1, "shannon_inchikey", 0.0)

# InChiKey regex definition
InChiKey_REGEX_PATTERN = r"""([A-Z])"""
regex = re.compile(InChiKey_REGEX_PATTERN)


def shannon_entropy_inch(ik):
    
    molecule = ik
    tokens = regex.findall(molecule)
    # print(tokens)

    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    import math
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)  
        
    return shannon  


shannon_arr = []
for p in range(0,len(df['shannon_inchikey'])):

    smiles = df['Smiles'][p]
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)
    m_inchkey = Chem.MolToInchiKey(mol)
    ik = m_inchkey.split("-")
    # print("InchiKey splitted", ik)
    # print("\n")

    shannon_entropy = 0
    for i in range(len(ik)):
    
        if i<=1:
            shannon_entropy_by_parts = shannon_entropy_inch(ik[i])
            shannon_entropy = shannon_entropy + shannon_entropy_by_parts  
            # print(shannon_entropy_by_parts)
        else:
            freq = 1/25 ### Inchikey contains total 25 characters, apart from 2 hyphens
            shannon_entropy_by_parts = - freq * math.log2(freq)
            shannon_entropy = shannon_entropy + shannon_entropy_by_parts  
            # print(shannon_entropy_by_parts)
        
        
    print("shannon entropy on inchikey: ", shannon_entropy)
    shannon_arr.append(shannon_entropy) 
    
df['shannon_inchikey']= shannon_arr 

###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    
print("[INFO] constructing training/ testing split")
split = train_test_split( df, test_size = 0.2, random_state = 42) 


# Distribute the input data columns in train & test splits
(XtrainTotalData, XtestTotalData) = split  # split format always is in (data_train, data_test, label_train, label_test)

# Normalize columns corresponding to max of (pChEMBL Value/MW) values
XtrainLabels  = XtrainTotalData.iloc[:,-1]/ (maxPrice)
XtestLabels  = XtestTotalData.iloc[:,-1]/ (maxPrice)   

 
# Just the 1st column (mol. wt.) & 2nd col (SMILES Shannon entropy for this case) as X data
XtrainData = (XtrainTotalData.iloc[:,0:2])
XtestData = (XtestTotalData.iloc[:,0:2])

print("XtrainData shape",XtrainData.shape)
print("XtestData shape",XtestData.shape)


# perform min-max scaling each continuous feature column to the range [0 1]
cs = MinMaxScaler()
trainContinuous = cs.fit_transform(XtrainTotalData.iloc[:,0:2])
testContinuous = cs.transform(XtestTotalData.iloc[:,0:2])


print("[INFO] processing input data after normalization....")
XtrainData, XtestData = trainContinuous,testContinuous ## Feeding single array as XtrainData and XtestData

# create the MLP model
mlp = KiNet_mlp.create_mlp(XtrainData.shape[1], regress = False) # the input dimension to mlp would be shape[1] of the matrix i.e. column features
print("shape of mlp", mlp.output.shape)

# Processing output of MLP model
combinedInput = mlp.output
print("shape of combinedInput",combinedInput.shape)

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
H = model.fit( x = XtrainData , y = trainY, validation_data = ( XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 


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
plt.savefig('pchembl_MW_pred_with_shannon.png')
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
plt.savefig('loss@epochs_pChEMBL_MW_with_shannon_MLP_only')
####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
