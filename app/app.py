#!flask/bin/python
from flask import Flask, abort, jsonify
from flask import request

from claim_subject import ClaimsSubject

app = Flask(__name__)
model_taxonomy = ClaimsSubject()


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/claim/api/v1.0/taxonomy', methods=['POST'])
def perform_claim():
    if not request.json or not 'body' in request.json:
        abort(400)
    text = request.json['body']
    outputs = model_taxonomy.perform(text)
    return jsonify(outputs), 201


if __name__ == '__main__':
    app.run(debug=True)
