from flask import Flask,render_template,request
from flask_material import Material

import numpy as np

import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cancer',methods=["POST","GET"])
def cancer():
    if request.method == 'POST':
        ct = request.form['ct']
        csz = request.form['csz']
        csp = request.form['csp']
        cad = request.form['cad']
        es = request.form['es']
        bn = request.form['bn']
        bc = request.form['bc']
        nn = request.form['nn']

        sample_data = [ct,csz,csp,cad,es,bn,bc,nn]
        clean_data = [float(i) for i in sample_data]
        
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('C:/Users/avisk/OneDrive/Desktop/TEST/data/cancer_model.pkl')
        prediction = load_model.predict(ex1)
        pred = load_model.predict([clean_data])
        return render_template('cancer.html',pred=pred)
    else:
        return render_template('cancer.html')


@app.route('/dibetes',methods=["POST","GET"])
def dibetes():
    if request.method == 'POST':
        preg = request.form['preg']
        glc = request.form['glc']
        bp = request.form['bp']
        st = request.form['st']
        ins = request.form['ins']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        sample_data = [preg,glc,bp,st,ins,bmi,dpf,age]
        clean_data = [float(i) for i in sample_data]
        
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('C:/Users/avisk/OneDrive/Desktop/TEST/data/dibetes_model.pkl')
        pred = load_model.predict(ex1)
        return render_template('dibetes.html',pred=pred)
    else:
        return render_template('dibetes.html')


@app.route('/kidney',methods = ["POST","GET"])
def kidney():
    if request.method == 'POST':
        pcv = request.form['pcv']
        sc = request.form['sc']
        sg = request.form['sg']
        su = request.form['su']
        age = request.form['age']
        pot = request.form['pot']
        wc = request.form['wc']
        rc = request.form['rc']
        al = request.form['al']
        bgr = request.form['bgr']
        dm = request.form['dm']

        sample_data = [pcv,sc,sg,su,age,pot,wc,rc,al,bgr,dm]
        clean_data = [float(i) for i in sample_data]
        
        print(type(clean_data))
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('C:/Users/avisk/OneDrive/Desktop/TEST/data/kidney_model.pkl')
        prediction = load_model.predict(ex1)
        print(prediction)
        pred = load_model.predict([clean_data])
        return render_template('kidney.html',pred=pred)
    else:
        return render_template('kidney.html')


@app.route('/heart',methods=["POST","GET"])
def heart():    
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cigspDay = request.form['cigspDay']
        bpmeds = request.form['bpmeds']
        totChol = request.form['totChol']
        sysbp = request.form['sysbp']
        gls = request.form['gls']

        sample_data = [age,sex,cigspDay,bpmeds,totChol,sysbp,gls]
        clean_data = [eval(i) for i in sample_data]
        
        print(type(clean_data))
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('C:/Users/avisk/OneDrive/Desktop/TEST/data/heart_model.pkl')
        prediction = load_model.predict(ex1)
        print(prediction)
        pred = load_model.predict([clean_data])
        return render_template('heart.html',pred=pred)
    else:
        return render_template('heart.html')

if __name__ == "__main__":
    app.run(port=8000)