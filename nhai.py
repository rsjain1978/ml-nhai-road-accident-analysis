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