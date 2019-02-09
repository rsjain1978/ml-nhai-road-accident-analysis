def removeUnrequiredFeature(data, featureToRemove):
    data=data.drop([featureToRemove],axis=1)
    return data