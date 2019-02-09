import pandas as pd
from sklearn.preprocessing import LabelEncoder

def helpProvidedDataPreProcessing (data):

    #Encode HelpProvidedByPoliceOrAmbulance
    #labelEncoder = LabelEncoder()
    #helpProvidedBy = data['HelpProvidedByAmbulancePatrol']
    #encoded_HelpProvidedBy = labelEncoder.fit_transform(helpProvidedBy)
    
    encoded_HelpProvidedBy = pd.get_dummies(data['HelpProvidedByAmbulancePatrol'],prefix='HelpProvidedBy')

    data['HelpProvidedBy_Ambulance']=encoded_HelpProvidedBy['HelpProvidedBy_Ambulance']
    data['HelpProvidedBy_Ambulance/Petrol Vehicle']=encoded_HelpProvidedBy['HelpProvidedBy_Ambulance/Petrol Vehicle']
    data['HelpProvidedBy_Petrol Vehicle']=encoded_HelpProvidedBy['HelpProvidedBy_Petrol Vehicle']

    data=data.drop(['HelpProvidedByAmbulancePatrol'],axis=1)

    return data