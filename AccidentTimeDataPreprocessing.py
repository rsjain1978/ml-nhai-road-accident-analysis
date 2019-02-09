import pandas as pd
from sklearn.preprocessing import LabelEncoder

def accidentTimePreProcessing(data):
    #From the AccTime column create another column to show if the time was in AM or PM
    newAccidentTimeCols = data["TimeOfAcc"].str.split(" ", n = 1, expand = True) 
    data['TimeOfAccNumeric']=newAccidentTimeCols[0]
    data['TimeOfAccAMPM']=newAccidentTimeCols[1]
    data=data.drop(['TimeOfAcc'],axis=1)

    accidentHourMinute = data['TimeOfAccNumeric'].str.split(":", n=1, expand=True)
    data['HourOfAccident'] = accidentHourMinute[0]

    data=data.drop(['TimeOfAccNumeric'],axis=1)

    #encode AM & PM time of acccident
    #labelEncoder = LabelEncoder()
    #data['Encoded_AccidentTimeMorningEvening']=labelEncoder.fit_transform(data['TimeOfAccAMPM'])
    time = pd.get_dummies(data['TimeOfAccAMPM'])
    data['TimeOfAccAMPM_AM']=time['AM']
    data['TimeOfAccAMPM_PM']=time['PM']

    data=data.drop(['TimeOfAccAMPM'],axis=1)

    return data