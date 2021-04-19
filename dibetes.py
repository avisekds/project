def dibetesFunction():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import preprocessing
    from sklearn import utils
    import joblib
    from sklearn.metrics import confusion_matrix
    import sklearn


    df=pd.read_csv("kaggle_diabetes.csv")

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    test = SelectKBest(score_func=chi2, k=4)




    X = df.iloc[:,:-1]
    y=df["Outcome"]


    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=50)




    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)



    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier


    knn = KNeighborsClassifier()
    dtree = DecisionTreeClassifier()
    svm = SVC()
    logit = LogisticRegression()
    xgb= XGBClassifier()

    bestAccur = {}

    logit.fit(x_train,y_train)
    y_pred=logit.predict(x_test)
    bestAccur["logit"] = [sklearn.metrics.accuracy_score(y_test,y_pred),logit]

    dtree.fit(x_train, y_train)
    y_pred=dtree.predict(x_test)
    bestAccur["dtree"] = [sklearn.metrics.accuracy_score(y_test,y_pred),dtree]

    knn.fit(x_train, y_train)
    y_pred=knn.predict(x_test)
    bestAccur["knn"] = [sklearn.metrics.accuracy_score(y_test,y_pred),knn]

    svm.fit(x_train, y_train)
    y_pred=svm.predict(x_test)
    bestAccur["svm"] = [sklearn.metrics.accuracy_score(y_test,y_pred),svm]

    xgb.fit(x_train, y_train)
    y_pred=xgb.predict(x_test)
    bestAccur["xgb"] = [sklearn.metrics.accuracy_score(y_test,y_pred),xgb]
        
    best = bestAccur["knn"][0]*1000
    predictor = bestAccur["knn"][1] 
    print(type(best),best)
    for ele in bestAccur:
        if int(bestAccur[ele][0]*1000)>int(best):
            best = bestAccur[ele][0]*1000
            predictor = bestAccur[ele][1]
    return predictor