from flask import Flask, render_template,url_for,request
from flask_material import Material


#EDA PKg
import pandas as pd
import numpy as np

#ML PKg
import sklearn.externals
import joblib
 
app = Flask(__name__)
Material(app)
 
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv('data/iris ds.csv')
    return render_template("preview.html", df_view=df)

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        petal_length = request.form['petal_length']
        sepal_length = request.form['sepal_length']
        petal_width = request.form['petal_width']
        sepal_width = request.form['sepal_width']
    

        arr = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
        #change from unicode to float
             
        svm_model = joblib.load('data/model_joblib')
        result_prediction = svm_model.predict(arr)
    
    return render_template("index.html", petal_width=petal_width,
        sepal_width=sepal_width,
        sepal_length=sepal_length, 
        petal_length=petal_length,
        svm_model=svm_model,
        result_prediction=result_prediction,
        )


if __name__=='__main__':
    app.run(debug=True)