This is a python implementation to classify accidents related data from NHAI for bangalore.

Main python file is nhai.py, you can execute that to see the classifier running. Classification prediction is 83%+. Output from run is as follows:

********************* Classifier Performance Report On Test Data ***********************
Feature Importance is                  importance
Fatal             0.235941
AccDay            0.168079
Grevious          0.147702
AccMonth          0.139050
Minor             0.138721
NatureAccident    0.099926
Injured           0.070581
Out of box features score is  0.6470588235294118
[[5 1 0 0 0]
 [1 3 0 1 0]
 [0 0 5 1 0]
 [0 0 0 2 0]
 [0 0 0 0 4]]
Accuracy score is 0.8260869565217391
Recall score is 0.8260869565217391
Precision store is 0.8586956521739131
F1 score is 0.8313570487483529
              precision    recall  f1-score   support

           1       0.83      0.83      0.83         6
           2       0.75      0.60      0.67         5
           3       1.00      0.83      0.91         6
           4       0.50      1.00      0.67         2
           6       1.00      1.00      1.00         4

   micro avg       0.83      0.83      0.83        23
   macro avg       0.82      0.85      0.82        23
weighted avg       0.86      0.83      0.83        23

Key Points are:
1. Model accuracy is 83%+
2. Top 5 features for Accident Type are:
    2.1 Fatal (Most of accidents had no fatalities)
    2.2 Day of Accident (Occurences of accidents are spread across all days of month, although they tend to happen more at the start of the month)
    2.3 Grevious
    2.4 Month of Accident
    2.5 Minor Accident or not (Most accidents are minor in nature)
    2.6 Nature of Accident (Most accidents are caused either by Overturning or as Head-on Collisions)
3. Following areas are most prone to accidents:
    3.1 126 KM on Right Hand Side
    3.2 115 KM on Right Hand Side
    3.3 126 KM on Left Hand Side
    3.4 109 KM on Right Hand Side
    3.5 Around 86 KM
