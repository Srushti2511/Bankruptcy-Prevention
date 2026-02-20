# test_app.py
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET'])
def index():
    app.logger.info("GET / received")
    return "Test app running. Use POST /predict to test."

@app.route('/predict', methods=['GET'])
def predict_get():
    app.logger.info("GET /predict received")
    return "This endpoint accepts POST JSON. Use POST to test."

@app.route('/predict', methods=['POST'])
def predict_post():
    data = request.get_json(force=True, silent=True)
    app.logger.info("POST /predict received with body: %s", data)
    # echo back the data to confirm it's received
    return jsonify({'received': data})

if __name__ == '__main__':
    app.run(debug=True)
