#!flask/bin/python
from flask import Flask, abort, jsonify
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/claim/api/v1.0/text', methods=['POST'])
def perform_claim():
    if not request.json or not 'body' in request.json:
        abort(400)
    text = request.json['body']
    return jsonify({'body': text}), 201

if __name__ == '__main__':
    app.run(debug=True)