import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def printClassifierPerformanceOnTrainData(rf, train_X, train_y, predicted_y_with_train_data):
    print ('    ')
    print ('********************* Classifier Performance Report On Training Data ***********************')

    printClassifierPerformance(rf, train_X, train_y, predicted_y_with_train_data)

def printClassifierPerformanceOnTestData(rf, train_X, test_y, predicted_y_with_test_data):
    print ('    ')
    print ('********************* Classifier Performance Report On Test Data ***********************')
    printClassifierPerformance(rf, train_X, test_y, predicted_y_with_test_data)

def printClassifierPerformance (rf, train_X, actualY, predictedY):

    feature_importances = pd.DataFrame(rf.feature_importances_,index = train_X.columns,columns=['importance']).sort_values('importance', ascending=False)

    #print the feature importance - tbd
    print ('Feature Importance is ',feature_importances)
                                        
    #print the oob-score (out of box features error score)
    print ('Out of box features score is ',rf.oob_score_)

    #print the confusion matrix
    c_matrix = confusion_matrix(actualY, predictedY)
    print (c_matrix)

    print ('Accuracy score is',accuracy_score(actualY, predictedY))

    print ('Recall score is', recall_score(actualY, predictedY, average='weighted'))

    print ('Precision store is', precision_score(actualY, predictedY, average='weighted'))

    print ("F1 score is", f1_score(actualY, predictedY, average='weighted'))

    #print the classification report
    print (classification_report(actualY, predictedY))