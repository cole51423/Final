from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

# Default network structure: 3 layers (input, hidden, output)
DEFAULT_STRUCTURE = [3, 4, 2]

def random_weights_biases(structure):
    weights = [np.random.randn(structure[i], structure[i-1]).tolist() for i in range(1, len(structure))]
    biases = [np.random.randn(n, 1).tolist() for n in structure[1:]]
    return weights, biases

def forward_pass(inputs, weights, biases):
    a = np.array(inputs).reshape(-1, 1)
    activations = [a.copy()]
    for w, b in zip(weights, biases):
        w = np.array(w)
        b = np.array(b)
        a = np.dot(w, a) + b
        a = 1 / (1 + np.exp(-a))  # sigmoid
        activations.append(a.copy())
    return activations

@app.route('/')
def index():
    return render_template('index.html', structure=DEFAULT_STRUCTURE)

@app.route('/forward', methods=['POST'])
def forward():
    data = request.json
    inputs = data['inputs']
    weights = data['weights']
    biases = data['biases']
    activations = forward_pass(inputs, weights, biases)
    output = activations[-1].flatten().tolist()
    # Return all activations for hidden layer visualization
    all_activations = [a.flatten().tolist() for a in activations]
    return jsonify({'output': output, 'activations': all_activations})

@app.route('/random-params', methods=['POST'])
def random_params():
    structure = request.json.get('structure', DEFAULT_STRUCTURE)
    weights, biases = random_weights_biases(structure)
    return jsonify({'weights': weights, 'biases': biases})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
