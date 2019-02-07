#Consider National Highway Authority of India (NHAI) provides you a real accidental data set of Bangalore highway of India. The size of data set is (83 x 8). Apply suitable Machine Learning technique to address following problems.
#1. Forecast Accident type(variable C in the data set) for year 2015 based on the observations given for year 2014 in the data set. Analyse the predictive results achieved.
#2. Use feature Location in the data set to identify the prone area of major accidents.
#3. Identify top 5 important features for variable Accident type(variable C in the data set)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


data = pd.read_csv('NHAIAccidentData.csv',dtype={})
print (data.shape)

#Convert the Accident Date column into Day, Month and Year column
accidentDate=pd.to_datetime(data['Date'])
data['AccYear']=accidentDate.dt.year
data['AccMonth']=accidentDate.dt.month
data['AccDay']=accidentDate.dt.day
data=data.drop(['Date'],axis=1)
print (data.shape)

#Encode HelpProvidedByPoliceOrAmbulance
labelEncoder = LabelEncoder()
helpProvidedBy = data['HelpProvidedByAmbulancePatrol']
encoded_HelpProvidedBy = labelEncoder.fit_transform(helpProvidedBy)
data['Encoded_HelpProvidedByPoliceAmbulance']=encoded_HelpProvidedBy
data=data.drop(['HelpProvidedByAmbulancePatrol'],axis=1)
print (data.shape)

#From the AccTime column create another column to show if the time was in AM or PM
# new data frame with split value columns 
newAccidentTimeCols = data["TimeOfAcc"].str.split(" ", n = 1, expand = True) 
data['TimeOfAccNumeric']=newAccidentTimeCols[0]
data['TimeOfAccAMPM']=newAccidentTimeCols[1]
data=data.drop(['TimeOfAcc'],axis=1)
data=data.drop(['TimeOfAccNumeric'],axis=1)
print(data.shape)

#encode AM & PM time of acccident
labelEncoder = LabelEncoder()
data['Encoded_AccidentTimeMorningEvening']=labelEncoder.fit_transform(data['TimeOfAccAMPM'])
data=data.drop(['TimeOfAccAMPM'],axis=1)
print(data.shape)

#Convert Vehicle Responsible into encoded values--FEEDBACK
#data['Encoded_VehicleResponsible'] = labelEncoder.fit_transform(data['VehicleResponsible'])
#data=data.drop(['VehicleResponsible'],axis=1)
#print(data.shape)

#Encode Vehicles Involved with a custom logic
data.loc [data['VehicleResponsible'].str.contains('lorry',case=False), 'Encoded_VehicleResponsible']='0'
data.loc [data['VehicleResponsible'].str.contains('truck',case=False), 'Encoded_VehicleResponsible']='0'
data.loc [data['VehicleResponsible'].str.contains('car',case=False), 'Encoded_VehicleResponsible']='1'
data.loc [data['VehicleResponsible'].str.contains('bus',case=False), 'Encoded_VehicleResponsible']='0'
data.loc [data['VehicleResponsible'].str.contains('motor cycle',case=False), 'Encoded_VehicleResponsible']='2'
data.loc [data['VehicleResponsible'].str.contains('two wheeler',case=False), 'Encoded_VehicleResponsible']='2'
data.loc [data['VehicleResponsible'].str.contains('bike',case=False), 'Encoded_VehicleResponsible']='2'
data.loc [data['VehicleResponsible'].str.contains('unknown',case=False), 'Encoded_VehicleResponsible']='3'
data.loc [data['VehicleResponsible'].str.contains('lcv',case=False), 'Encoded_VehicleResponsible']='1'
data.loc [data['VehicleResponsible'].str.contains('tipper',case=False), 'Encoded_VehicleResponsible']='1'
data.loc [data['VehicleResponsible'].str.contains('matador',case=False), 'Encoded_VehicleResponsible']='1'
data.loc [data['VehicleResponsible'] =='', 'Encoded_VehicleResponsible']='4'

data=data.drop(['VehicleResponsible'],axis=1)
print(data.shape)

#Where Accident Classification is  '-' encode it to some value say 5
data.loc[data['ClassificationOfAccident']=='-', 'ClassificationOfAccident'] = '6'

#Remove the remakrs column
data=data.drop(['Remarks'],axis=1)

#Remove the Accidents location column
data=data.drop(['AccLocation'],axis=1)

# dropping null value columns to avoid errors 
#data.loc[data['Causes']=='-', 'Causes'] = 'NaN'
data = data[~data['Causes'].str.contains('-')]
#data.dropna(inplace = True) 

#After run 1 this feature was found to have least impact
data=data.drop(['Grevious'],axis=1)

#After run 2 this feature was found to have least impact
data=data.drop(['Minor'],axis=1)

#Y1 =  Y.drop(Y[Y['ClassificationOfAccident']=='-'].index)
#print (Y1['ClassificationOfAccident'].value_counts())

data.to_csv('FormattedNHAIAccidentsData.csv')

#Read column 3 which is about 'ClassificationOfAccident' as class variable
Y=data['ClassificationOfAccident']
X=data.drop(['ClassificationOfAccident'],axis=1)

data.to_csv('FeaturesData.csv')

################################################
############# Feature Selection#################
################################################

print ("**** Model performance before feature selection ******")
train_x, test_x, train_y, test_y = train_test_split(X,Y,random_state=1)

#initialize a Random forest classifier with 
# 1000 decision trees or estimators
# criteria as entropy, 
# max depth of decision trees as 10
# max features in each decision tree be selected automatically
rf = RandomForestClassifier(n_estimators=1000,
        max_depth=10, 
        max_features='auto', 
        bootstrap=True,
        oob_score=True,
        random_state=1)

#fit the data        
rf.fit(train_x, train_y)

#print the feature importance - tbd
print ('Feature Importance is ',rf.feature_importances_)

#print the oob-score (out of box features error score)
print ('Out of box features score is ',rf.oob_score_)

#do a prediction on the test X data set
predicted_y = rf.predict(test_x)

#errors = abs(predicted_y-test_y)
#print ('Mean absolute error (MAE) ', round(np.mean(errors),2))

#print the confusion matrix
confusion_matrix = confusion_matrix(test_y, predicted_y)
print (confusion_matrix)

print ('Accuracy score is',accuracy_score(test_y, predicted_y))

print ('Recall score is', recall_score(test_y, predicted_y, average='weighted'))

print ('Precision store is', precision_score(test_y, predicted_y, average='weighted'))

print ("F1 score is", f1_score(test_y, predicted_y, average='weighted'))

#print the classification report
print (classification_report(test_y, predicted_y))