from flask import Flask,request, render_template
import numpy as np
import joblib

#loading models
knn = joblib.load('./static/knn.joblib')

preprocessor = joblib.load('./static/pre.joblib')

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/main')
def main():
    return render_template('main.html')



@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        start = request.form['start']
        end = request.form['end']
        area= request.form['area']
        season = request.form['season']
        state = request.form['state']
        district  = request.form['district']
        features = np.array([[state,district,season,start,end,area]],dtype=object)
        print(features)
        transformed_features = preprocessor.transform(features)
        prediction = knn.predict(transformed_features)
        
        prediction_value = float(prediction[0])
        area_value = float(area)
        yield_value = prediction_value / area_value
        yield_value=round(yield_value, 2)
        return render_template('main.html',prediction = 'The Predicted Production is {}tonnes and Yield is {}tonnes/hectare'.format(prediction[0],yield_value))

if __name__=="__main__":
    app.run(debug=True)