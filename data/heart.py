def heartFunction():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sn
    import joblib
    from sklearn.metrics import confusion_matrix

    heart_df=pd.read_csv("heart.csv")
    heart_df.drop(columns=['education'],inplace=True)
    heart_df.rename(columns={'male':'Gender'},inplace=True)
    heart_df.dropna(axis=0,inplace=True)


    import sklearn
    x=heart_df[['age','Gender','cigsPerDay','BPMeds','totChol','sysBP','glucose']]
    y=heart_df['TenYearCHD']
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
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