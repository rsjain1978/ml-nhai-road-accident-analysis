#Consider National Highway Authority of India (NHAI) provides you a real accidental data set of Bangalore highway of India. The size of data set is (83 x 8). Apply suitable Machine Learning technique to address following problems.
#1. Forecast Accident type(variable C in the data set) for year 2015 based on the observations given for year 2014 in the data set. Analyse the predictive results achieved.
#2. Use feature Location in the data set to identify the prone area of major accidents.
#3. Identify top 5 important features for variable Accident type(variable C in the data set)

import pandas as pd

data = pd.read_csv('NHAIAccidentData.csv')
print (data.shape)

#Read column 3 which is about 'ClassificationOfAccident' as class variable
Y=data.iloc[:,4:5]
print (Y)
print (Y.shape)
print (Y['ClassificationOfAccident'].value_counts())
Y1 =  Y.drop(Y[Y['ClassificationOfAccident']=='-'].index)
print (Y1)
print (Y1.shape)
print (Y1['ClassificationOfAccident'].value_counts())

#some of the class variables are blank, check the count

#Remove class variable from X
X=data.drop(['ClassificationOfAccident'],axis=1)
print (X.shape)

i=1
while i<17:
    print (X.iloc[:,i-1:i])
    i=i+1