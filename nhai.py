#Consider National Highway Authority of India (NHAI) provides you a real accidental data set of Bangalore highway of India. The size of data set is (83 x 8). Apply suitable Machine Learning technique to address following problems.
#1. Forecast Accident type(variable C in the data set) for year 2015 based on the observations given for year 2014 in the data set. Analyse the predictive results achieved.
#2. Use feature Location in the data set to identify the prone area of major accidents.
#3. Identify top 5 important features for variable Accident type(variable C in the data set)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
print(data.shape)

#encode AM & PM time of acccident
labelEncoder = LabelEncoder()
data['Encoded_AccidentTimeMorningEvening']=labelEncoder.fit_transform(data['TimeOfAccAMPM'])
data=data.drop(['TimeOfAccAMPM'],axis=1)
print(data.shape)

#Convert Vehicle Responsible into encoded values
data['Encoded_VehicleResponsible'] = labelEncoder.fit_transform(data['VehicleResponsible'])

data.to_csv('FormattedNHAIAccidentsData.csv')


#Read column 3 which is about 'ClassificationOfAccident' as class variable
#Y=data.iloc[:,4:5]
#Y1 =  Y.drop(Y[Y['ClassificationOfAccident']=='-'].index)
#print (Y1['ClassificationOfAccident'].value_counts())

#some of the class variables are blank, check the count

#Remove class variable from X
X=data.drop(['ClassificationOfAccident'],axis=1)
print (X.shape)

#Print each column of the dataframe
i=1
while i<17:
    print (X.iloc[:,i-1:i])
    i=i+1