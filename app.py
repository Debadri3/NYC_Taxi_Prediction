import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('nyc_xgbreg.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data1 = int(request.form['ft_5'])
    data2 = int(request.form['ft_4'])
    data3 = int(request.form['ft_3'])
    data4 = int(request.form['ft_2'])
    data5 = int(request.form['ft_1'])
    data6 = float(request.form['lat'])
    data7 = float(request.form['lon'])
    data8=  int(request.form['weekday'])
    data9=  int(request.form['exp_avg'])
    final_features = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9]])
    prediction = model.predict(final_features)

    output = round(prediction[0], 0)

    return render_template('index.html', prediction_text='The number of pickups should should be around {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)