#Consider National Highway Authority of India (NHAI) provides you a real accidental data set of Bangalore highway of India. The size of data set is (83 x 8). Apply suitable Machine Learning technique to address following problems.
#1. Forecast Accident type(variable C in the data set) for year 2015 based on the observations given for year 2014 in the data set. Analyse the predictive results achieved.
#2. Use feature Location in the data set to identify the prone area of major accidents.
#3. Identify top 5 important features for variable Accident type(variable C in the data set)

import pandas as pd
import numpy as np

import VehiclesDataPreprocessing as vdp
import AccidentDateDataPreprocessing as adp
import HelpProvidedDataPreprocessing as hdp
import AccidentTimeDataPreprocessing as atdp
import DataPreProcessingUtils as dpu
import PrintClassifierPerformance as pcp

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('NHAIAccidentData.csv',dtype={})
print ('Data shape before pre-processing-',data.shape)

data = adp.preProcessAccidentDateData(data)
data = hdp.helpProvidedDataPreProcessing(data)
data = atdp.accidentTimePreProcessing(data)
data = vdp.preProcessResponsibleVehiclesData(data)

#Where Accident Classification is  '-' encode it to some value say 5
data.loc[data['ClassificationOfAccident']=='-', 'ClassificationOfAccident'] = '6'

data = dpu.removeUnrequiredFeature(data, 'Remarks')
data = dpu.removeUnrequiredFeature(data, 'AccLocation')
# data = dpu.removeUnrequiredFeature(data, 'Grevious')
# data = dpu.removeUnrequiredFeature(data, 'Minor')

##### Took these attributes out since the visualization proved that they have skewed values 
data = dpu.removeUnrequiredFeature(data, 'WeatherCondition')
data = dpu.removeUnrequiredFeature(data, 'NumAnimalsKilled')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_3')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_1')
data = dpu.removeUnrequiredFeature(data, 'HelpProvidedBy_Ambulance/Petrol Vehicle')
data = dpu.removeUnrequiredFeature(data, 'HelpProvidedBy_Petrol Vehicle')
data = dpu.removeUnrequiredFeature(data, 'HelpProvidedBy_Ambulance')
data = dpu.removeUnrequiredFeature(data, 'IntersectionTypeControl')
data = dpu.removeUnrequiredFeature(data, 'AccYear')
data = dpu.removeUnrequiredFeature(data, 'RoadCondition')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_2')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_0')
#data = dpu.removeUnrequiredFeature(data, 'Encoded_AccidentTimeMorningEvening')
data = dpu.removeUnrequiredFeature(data,'TimeOfAccAMPM_AM')
data = dpu.removeUnrequiredFeature(data,'TimeOfAccAMPM_PM')
data = dpu.removeUnrequiredFeature(data, 'Causes')
data = dpu.removeUnrequiredFeature(data, 'RoadFeature')
data = dpu.removeUnrequiredFeature(data, 'HourOfAccident')
#data = dpu.removeUnrequiredFeature(data, 'Injured')
#data = dpu.removeUnrequiredFeature(data, 'NatureAccident')

# dropping null value columns to avoid errors 
#data = data[~data['Causes'].str.contains('-')]

print ('Data shape after pre-processing-',data.shape)

data.to_csv('FormattedNHAIAccidentsData.csv')

#Read column 3 which is about 'ClassificationOfAccident' as class variable
Y=data['ClassificationOfAccident']
X=data.drop(['ClassificationOfAccident'],axis=1)

data.to_csv('FeaturesData.csv')

################################################
############# Feature Selection#################
################################################

train_X, test_x, train_y, test_y = train_test_split(X,Y,random_state=1)

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
rf.fit(train_X, train_y)

predicted_y_with_train_data = rf.predict(train_X)
pcp.printClassifierPerformanceOnTrainData(rf, train_X, train_y, predicted_y_with_train_data)


rf = RandomForestClassifier(n_estimators=1000,
        max_depth=10, 
        max_features='auto', 
        bootstrap=True,
        oob_score=True,
        random_state=1)

#fit the data        
rf.fit(train_X, train_y)

#do a prediction on the test X data set
predicted_y_with_test_data = rf.predict(test_x)
pcp.printClassifierPerformanceOnTestData(rf, train_X, test_y, predicted_y_with_test_data)