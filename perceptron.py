import numpy as np


def sigmoid(x):
	return 1/(1+ np.exp(-x))

def sigmoid_der(x):
	return x*(1-x)

t_input = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

t_output = np.array([[0,1,1,0]]).T
np.random.seed(1)

synaptic_weights = 2*np.random.random((3,1))-1

print('random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(60000):

	input_layer = t_input
	output = sigmoid(np.dot(input_layer, synaptic_weights))
	error = t_output - output
	adj = error*sigmoid_der(output)
	synaptic_weights += np.dot(input_layer.T, adj)

print('synaptic weights after testing ')
print(synaptic_weights)

print('output after teaing:')
print(output)