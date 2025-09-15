import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask,render_template,request

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/',methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        features = []
        features.append(int(request.form['lead_time']))
        features.append(int(request.form['no_of_special_request']))
        features.append(float(request.form['avg_price_per_room']))
        features.append(int(request.form['arrival_month']))
        features.append(int(request.form['arrival_date']))
        features.append(int(request.form['market_segment_type']))
        features.append(int(request.form['no_of_week_nights']))
        features.append(int(request.form['no_of_weekend_nights']))
        features.append(int(request.form['type_of_meal_plan']))
        features.append(int(request.form['room_type_reserved']))

        features = np.array([features])
        prediction = loaded_model.predict(features)
        return render_template('index.html',prediction=prediction[0])
    return render_template('index.html',prediction=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=8080)
