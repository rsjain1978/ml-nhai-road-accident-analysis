import pandas as pd

def preProcessAccidentDateData(data):

    #Convert the Accident Date column into Day, Month and Year column
    accidentDate=pd.to_datetime(data['Date'])
    data['AccYear']=accidentDate.dt.year
    data['AccMonth']=accidentDate.dt.month
    data['AccDay']=accidentDate.dt.day

    #monthWise = pd.get_dummies(data['AccMonth'],prefix='Months')

    #data['Months_1']=monthWise['Months_1']
    #data['Months_2']=monthWise['Months_2']
    #data['Months_3']=monthWise['Months_3']
    #data['Months_4']=monthWise['Months_4']
    #data['Months_5']=monthWise['Months_5']
    #data['Months_6']=monthWise['Months_6']
    #data['Months_7']=monthWise['Months_7']
    #data['Months_8']=monthWise['Months_8']
    #data['Months_9']=monthWise['Months_9']
    #data['Months_10']=monthWise['Months_10']
    #data['Months_11']=monthWise['Months_11']
    #data['Months_12']=monthWise['Months_12']

    #data=data.drop(['AccMonth'],axis=1)
    data=data.drop(['Date'],axis=1)

    return data
