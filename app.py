from flask import Flask, render_template, request
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
app = Flask(__name__)
 
 
@app.route('/')
def index():
    return render_template('index.html')
 
 
 
@app.route('/predict',methods=['POST'])
def predict():
 
    if request.method == 'POST':
 
        CID = request.form['CID']
        Gen = request.form['Gen']
        Sc = request.form['Sc']
        Par = request.form['Par']
        Dep = request.form['Dep']
        Ten = request.form['Ten']
        PhnS = request.form['PhnS']
        MpL = request.form['MpL']
        InS = request.form['InS']
        OnS = request.form['OnS']
        OnB = request.form['OnB']
        DevP = request.form['DevP']
        TecS = request.form['TecS']
        Stv = request.form['Stv']
        Stm = request.form['Stm']
        Con = request.form['Con']
        Pb = request.form['Pb']
        MC = request.form['MC']
        data =[[int(CID),int(Gen),int(Sc),int(Par),int(Dep),int(Ten),int(PhnS),int(MpL),int(InS),int(OnS),int(OnB),int(DevP),int(TecS),int(Stv),int(Stm),int(Con),int(Pb),int(MC)]]
 
        lr_model = pickle.load(open('telco.pkl', 'rb'))
        prediction = lr_model.predict(data)[0]
 
    return render_template('index.html', prediction_text='Probability that the customer continues with the company is :{}'.format(prediction))
 
 
 
if __name__ == '__main__':
    app.run(debug=True)