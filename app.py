import pickle
import jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split




data=pd.read_csv('wine.csv')

X=data.drop(['quality'],axis=1)
y=data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
dt_model = DecisionTreeClassifier(criterion='entropy',max_depth=6,max_features='sqrt',max_leaf_nodes=10,min_samples_leaf=10,min_samples_split=2,random_state=10)
dt_model.fit(X_train, y_train)


filename = 'finalized_model.pkl'
pickle.dump(dt_model, open(filename, 'wb'))


app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('miniProject.html')
@app.route('/winequality.html')
def winequality():
    return render_template('winequality.html')
@app.route('/miniProject.html')
def backhome():
    return render_template('miniProject.html')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        fixed_acidity=float(request.form['fixed_acidity'])
        volatile_acidity=float(request.form['volatile_acidity'])
        citric_acid=float(request.form['citric_acid'])
        residual_sugar=float(request.form['residual_sugar'])
        chlorides=float(request.form['chlorides'])
        total_sulfur_dioxide=float(request.form['total_sulfur_dioxide'])
        density=float(request.form['density'])
        pH=float(request.form['pH'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])
        sulfur_dioxide_ratio=float(request.form['sulfur_dioxide_ratio'])
        type1=request.form['type1']
        if(type1=='Red'):             
            type1=1 	
        elif(type1=='White'):
            type1=0

        def lr(type1,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,total_sulfur_dioxide,density,pH,sulphates,alcohol,sulfur_dioxide_ratio):
            c=pd.DataFrame([type1,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,total_sulfur_dioxide,density,pH,sulphates,alcohol,sulfur_dioxide_ratio]).T
            return model.predict(c)
          
    
    prediction=lr(type1,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,total_sulfur_dioxide,density,pH,sulphates,alcohol,sulfur_dioxide_ratio)
    if prediction==0:
        return render_template('winequality.html',prediction_text="Wine Quality is Low")
    else:
        return render_template('winequality.html',prediction_text="Wine Quality is High")
    #return render_template('winequality.html',prediction_text="Wine Quality is {}".format(prediction))
  

if __name__=="__main__":
    app.run(debug=True)

