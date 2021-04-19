def cancerFunction():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix


    df=pd.read_csv("Cancer.csv")

    X=df[["Id","Cell Thickness","Cell Size","Cell Shape","Cell Adhesion","Epith Size","Bare Nuclei","Blood Cromatin","Normal Nucleoli","Mitoses"]]
    y=df[["Class"]]


    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=50)

    outcome_corr=df.corr().iloc[:,-1].values.tolist()[:-1]

    outcome_corr.sort(reverse=True)
    meancrr=sum(outcome_corr)/len(outcome_corr)

    allowedcol=[]
    for i in outcome_corr:
        if i>meancrr:
            allowedcol.append(df.corr().iloc[:,-1].values.tolist()[:-1].index(i))

    fin=[]
    col=df.columns.tolist()
    for i in col:
        if col.index(i) in allowedcol:
            fin.append(col[col.index(i)])
            
    df1=df[fin]

    df2=df1.fillna(df1.mean())

    X1=df2[["Cell Thickness","Cell Size","Cell Shape","Cell Adhesion","Epith Size","Bare Nuclei","Blood Cromatin","Normal Nucleoli"]]
    y1=df[["Class"]]

    from sklearn.model_selection import train_test_split
    X1_train,x1_test,Y1_train,y1_test=train_test_split(X1,y1,test_size=0.4,random_state=50)


    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    import sklearn

    knn = KNeighborsClassifier()
    dtree = DecisionTreeClassifier()
    svm = SVC()
    logit = LogisticRegression()
    xgb= XGBClassifier()

    bestAccur = {}

    logit.fit(X1_train,Y1_train)
    y_pred=logit.predict(x1_test)
    bestAccur["logit"] = [sklearn.metrics.accuracy_score(y1_test,y_pred),logit]

    dtree.fit(X1_train, Y1_train)
    y_pred=dtree.predict(x1_test)
    bestAccur["dtree"] = [sklearn.metrics.accuracy_score(y1_test,y_pred),dtree]

    knn.fit(X1_train, Y1_train)
    y_pred=knn.predict(x1_test)
    bestAccur["knn"] = [sklearn.metrics.accuracy_score(y1_test,y_pred),knn]

    svm.fit(X1_train, Y1_train)
    y_pred=svm.predict(x1_test)
    bestAccur["svm"] = [sklearn.metrics.accuracy_score(y1_test,y_pred),svm]

    xgb.fit(X1_train, Y1_train)
    y_pred=xgb.predict(x1_test)
    bestAccur["xgb"] = [sklearn.metrics.accuracy_score(y1_test,y_pred),xgb]
    

    best = bestAccur["knn"][0]*1000
    predictor = bestAccur["knn"][1] 
    print(type(best),best)
    for ele in bestAccur:
        if int(bestAccur[ele][0]*1000)>int(best):
            best = bestAccur[ele][0]*1000
            predictor = bestAccur[ele][1]

    return predictor

