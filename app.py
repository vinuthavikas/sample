from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def home():
    return render_template('index.html')

@app.route('/predict_file', methods=['POST'])
def predict_file():
    features = [[float(request.form[f]) for f in [ "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]]
    prediction = model.predict(features)
    return render_template('index.html', prediction_text=f'Prediction: {int(prediction[0])}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
