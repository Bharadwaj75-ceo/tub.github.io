from cProfile import run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from flask import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/view')
def view():
    dataset = pd.read_csv('Features Extraction File.csv')
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/model', methods=["POST","GET"])
def model():
    if request.method=="POST":
        global model, x_train,x_test,y_train,y_test

        df = pd.read_csv('Features Extraction File.csv')
        df.head()
        df = df[['spectral_centroid', 'spectral_bandwidth', 'rolloff', 'mfcc1', 'mfcc2',
       'mfcc3', 'mfcc5', 'mfcc6', 'mfcc8', 'mfcc12', 'mfcc14', 'mfcc21',
       'mfcc30', 'mfcc32', 'mfcc34', 'mfcc36','label']]
        df.head()
        x = df.drop(['label'], axis=1)
        y = df['label']
        x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, stratify=y, random_state=100)


        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s==1:
            from sklearn.linear_model import LogisticRegression
            logistc=LogisticRegression()
            logistc.fit(x_train,y_train)
            y_pred=logistc.predict(x_test)
            ac_logistc=accuracy_score(y_pred,y_test)
            ac_logistc=ac_logistc*100
            msg="The accuracy obtained by LogisticRegression is "+str(ac_logistc) + str('%')
            return render_template("model.html",msg=msg)
        elif s==2:
            from sklearn.svm import LinearSVC
            gnb=LinearSVC()
            gnb.fit(x_train,y_train)
            y_pred=gnb.predict(x_test)
            ac_gnb=accuracy_score(y_pred,y_test)
            ac_gnb=ac_gnb*100
            msg="The acuuracy obtained by Support Vector Classifier is  " +str(ac_gnb) + str("%")
            return render_template("model.html",msg=msg)
        elif s==3:
            from sklearn.tree import DecisionTreeClassifier
            dts=DecisionTreeClassifier()
            dts.fit(x_train,y_train)
            y_pred=dts.predict(x_test)
            ac_dts=accuracy_score(y_pred,y_test)
            ac_dts=ac_dts*100
            msg="The accuracy obtained by Decision tree Clasiifier "+str(ac_dts) +str('%')
            return render_template("model.html",msg=msg)
        elif s==4:
            from sklearn.ensemble import RandomForestClassifier
            rf=RandomForestClassifier()
            rf.fit(x_train,y_train)
            y_pred=rf.predict(x_test)
            ac_rf=accuracy_score(y_pred,y_test)
            ac_rf=ac_rf*100
            msg="The accuracy obtained by Randomforest Classifier is "+str(ac_rf) +str('%')
            return render_template("model.html",msg=msg)
        
        
    return render_template("model.html")


@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        f1=float(request.form['f1'])
        f2=float(request.form['f2'])
        f3=float(request.form['f3'])
        f4=float(request.form['f4'])
        f5=float(request.form['f5'])
        f6=float(request.form['f6'])
        f7=float(request.form['f7'])
        f8=float(request.form['f8'])
        f9=float(request.form['f9'])
        f10=float(request.form['f10'])
        f11=float(request.form['f11'])
        f12=float(request.form['f12'])
        f13=float(request.form['f13'])
        f14=float(request.form['f14'])
        f15=float(request.form['f15'])
        f16=float(request.form['f16'])

        lee=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]
        print(lee)

        import pickle
        model=DecisionTreeClassifier()
        model.fit(x_train,y_train)
        result=model.predict([lee])
        print(result)
        result = result[0]
        if result == 'covid':
            msg = 'The Person Has the Covid Disease, Please Consult With A Doctor 😲'
            return render_template('prediction.html', msg=msg)
        else:
            msg = 'You Don Not Have Covid, Enjoy Your Day 😜'
            return render_template('prediction.html', msg=msg)
        
    return render_template('prediction.html')

if __name__ == "__main__":
    app.run(debug=True)