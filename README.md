This is a python implementation to classify accidents related data from NHAI for bangalore. Main python file is ***Classifier.py***, you can execute that to see the classifier running. 

Output from run is as follows:

  

***Classifier Performance Report On Test Data ***

Feature Importance is                  

Fatal             0.323514

Grevious          0.196601

Minor             0.176262

NatureAccident    0.168512

AccDay            0.135111

[[ 7  1  0  0]

[ 2 13  2  0]

[ 0  4  9  0]

[ 0  4  1  3]]

Accuracy score is 0.6956521739130435

Recall score is 0.6956521739130435

Precision store is 0.7395147123407994

F1 score is 0.6879392389366814
              precision    recall  f1-score   support

           1       0.78      0.88      0.82         8
           2       0.59      0.76      0.67        17
           3       0.75      0.69      0.72        13
           4       1.00      0.38      0.55         8

   micro avg       0.70      0.70      0.70        46

macro avg       0.78      0.68      0.69        46

weighted avg       0.74      0.70      0.69        46

**Sailient Features of Classifier are:**

 - Model accuracy is 69%+
 - Top 5 features for Accident Type are as follows:
   - Fatal
     - Most of accidents had no fatalities)
   - Grevious
   - Minor Accident 
     - Most accidents are minor in nature)	 
   - Nature of Accident 
     - Most accidents are caused either by Overturning or as Head-on Collisions)
   - Day of Accident 
     - Occurences of accidents are spread across all days of month, although they tend to happen more at the start of the month

	 
 - On doing data visualization following areas are most prone to accidents:
	 - 126 KM on Right Hand Side
	 - 115 KM on Right Hand Side
	 - 126 KM on Left Hand Side
	 - 109 KM on Right Hand Side
	 - Around 86 KM
