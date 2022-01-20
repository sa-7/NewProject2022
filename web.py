# importing libraries
from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import datetime as dt

# creating instance of Flask
app=Flask(__name__)

# opening and reading the pickle files created for each target field
model_newdeaths = pickle.load(open('model_new_deaths.pkl','rb'))
model_newcases = pickle.load(open('model_new_cases.pkl','rb'))
model_totalcases = pickle.load(open('model_total_cases.pkl','rb'))
model_totaldeaths = pickle.load(open('model_total_deaths.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/listing')
def listing():
    return render_template('result.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/predict',methods=['POST'])
# reading the values from screen and calling the model to predict the count. 
# After the count is predicted, value is displayed in result.html
def predict():
    country = request.form['country']
    select = request.form['radio']
    date1 = request.form['date']
    # formatting date column
    date2 = pd.to_datetime(date1).value
    xvalues = [date2, country]
    xvalues1 = [float(i) for i in xvalues]                    # converting values to float
    x_test = np.array(xvalues1).reshape(1, -1)                # converting to two dimensional array
    if select == 'total_cases':
        prediction = model_totalcases.predict(x_test)         # making prediction for total cases
    elif select == 'new_cases':
        prediction = model_newcases.predict(x_test)           # making prediction for new cases
    elif select == 'total_deaths':
        prediction = model_totaldeaths.predict(x_test)        # making prediction for total deaths
    elif select == 'new_deaths':
        prediction = model_newdeaths.predict(x_test)          # making prediction for new death
     
  
    return render_template ('result.html',prediction_text="The count is {}".format(prediction))
if __name__=='__main__':
    app.run(port=5000)

