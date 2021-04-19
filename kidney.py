def kidneyFunction():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import preprocessing
    from sklearn import utils
    import joblib
    from sklearn.metrics import confusion_matrix
    import sklearn

    df=pd.read_csv("kidney_disease.csv")
    df["rbc"].fillna("normal", inplace=True)
    df["pc"].fillna("normal", inplace=True)
    df["pcc"].fillna("notpresent", inplace=True)
    df["ba"].fillna("notpresent", inplace=True)
    df["htn"].fillna("no", inplace=True)
    df["dm"].fillna("no", inplace=True)
    df["cad"].fillna("no", inplace=True)
    df["appet"].fillna("good", inplace=True)
    df["pe"].fillna("no", inplace=True)
    df["classification"].fillna("ckd", inplace=True)

    df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
    df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
    df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
    df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
    df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
    df['appet'] = df['appet'].replace(to_replace='no',value=0)
    df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

    df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1,'no':0 ,'':np.nan})
    df['classification'] = df['classification'].replace(to_replace={'ckd':1,'ckd\t':1,'notckd':0,'no':0})


    df["rbc"]=df["rbc"].astype("category").cat.codes
    df["pc"]=df["pc"].astype("category").cat.codes
    df["pcc"]=df["pcc"].astype("category").cat.codes
    df["ba"]=df["ba"].astype("category").cat.codes
    df["htn"]=df["htn"].astype("category").cat.codes
    df["dm"]=df["dm"].astype("category").cat.codes
    df["cad"]=df["cad"].astype("category").cat.codes
    df["appet"]=df["appet"].astype("category").cat.codes
    df["pe"]=df["pe"].astype("category").cat.codes
    df["ane"]=df["ane"].astype("category").cat.codes

    df["pcv"]=pd.to_numeric(df['pcv'],errors='coerce')
    df["rc"]=pd.to_numeric(df['pcv'],errors='coerce')
    df["wc"]=pd.to_numeric(df['wc'],errors='coerce')


    df1=df.fillna(df.mean())
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    test = SelectKBest(score_func=chi2, k=4)
    X=df1[["age","sg","al","su","bgr","sc","pot","pcv","wc","rc","dm"]]
    y=df1['classification']

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=50)


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
    ran=RandomForestClassifier()
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


    best = bestAccur["knn"][0]*1000
    predictor = bestAccur["knn"][1] 
    for ele in bestAccur:
        if int(bestAccur[ele][0]*1000)>int(best):
            best = bestAccur[ele][0]*1000
            predictor = bestAccur[ele][1]
    # print(best)
    return predictor
