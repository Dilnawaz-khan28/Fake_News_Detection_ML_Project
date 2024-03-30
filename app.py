from flask import Flask, render_template, request
from NewsDetector import train_model, predict

app = Flask(__name__)

# Load the model when the Flask app starts
model = train_model('data/True.csv', 'data/Fake.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_news():
    user_input = request.form['news_input']
    prediction = predict(model, user_input)

    if prediction == 0:
        result = "The news is likely to be true."
    else:
        result = "The news is likely to be fake."

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
