from flask import Flask, render_template, request
import datetime
import time
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure the .pkl file is in the specified path)
model = pickle.load(open('Models/diabetes_nb_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Collect form data
    name = request.form['name']
    age = request.form['age']
    sex = request.form['sex']
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    bloodpressure = float(request.form['bloodpressure'])
    skinthickness = float(request.form['skinthickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetespedigreefunction = float(request.form['diabetespedigreefunction'])

    # Prepare the data for the model
    input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction]])
    
    # Start testing time
    start_time = time.time()

    # Predict using the loaded model
    prediction = model.predict(input_data)[0]

    # End testing time
    end_time = time.time()
    testing_time = round(end_time - start_time, 2)

    # Get current date and time
    report_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Map prediction result to a readable format
    prediction_result = "Positive" if prediction == 1 else "Negative"  # Update to "Positive" and "Negative" as per your UI

    # Render the result page with all the collected data
    return render_template('result.html', 
                           name=name, 
                           age=age, 
                           sex=sex,
                           pregnancies=pregnancies,
                           glucose=glucose,
                           bloodpressure=bloodpressure,
                           skinthickness=skinthickness,
                           insulin=insulin,
                           bmi=bmi,
                           diabetespedigreefunction=diabetespedigreefunction,
                           prediction=prediction_result,
                           report_datetime=report_datetime,
                           testing_time=testing_time)

if __name__ == '__main__':
    app.run(debug=False)
