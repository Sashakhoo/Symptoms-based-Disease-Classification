import numpy as np
import pickle
from flask import Flask, request, render_template

#Load ML model 
model = pickle.load(open('model.pkl','rb'))

#create application
app = Flask(__name__)

#Bind home function to URL
# Bind home function to URL
@app.route('/')
def home():
    return render_template('questions_test.html')

@app.route('/predict', methods =['POST'])
def predict():

    #put all entries in a list 
    features = [float(i) for i in request.form.values()]

    #Convert features to array
    array_features = [np.array(features)]

    #Predict features
    prediction = model.predict(array_features)
 
    return render_template('questions_test.html', result = prediction)


if __name__ =='__main__':
    #Run app
    app.run(debug=True)