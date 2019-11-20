import numpy as np
import os.path

class AI():
    def __init__(self, w1name, w2name):
        self.weights_one_name = w1name
        self.weights_two_name = w2name

        self.weights_one = []
        self.weights_two = []
        self.hidden_layer_size = 500
        self.input_layer_size = 4400

        self.score = 0
        self.average = .5
        self.ave_count = 1

        if os.path.exists(self.weights_one_name) and os.path.exists(self.weights_two_name):
            self.load_weights()

    def load_weights(self):
        self.weights_one = np.loadtxt(self.weights_one_name, delimiter=',')
        self.weights_two = np.loadtxt(self.weights_two_name, delimiter=',')

    def getAction(self,rgb, paddleA, paddleB, ball, reward, done):
        input_layer = self.screen_process(rgb)

        hidden_unactivated = np.dot(self.weights_one, input_layer)
        hidden = relu(hidden_unactivated)

        output_unactivated = np.dot(self.weights_two, hidden)
        output = relu(output_unactivated)

        self.score += reward
        self.average = ((self.average * self.ave_count) + output) / (self.ave_count + 1)
        self.ave_count += 1

        if output <= self.average:
            return 5
        else:
            return -5

    def screen_process(self, np_array):
        np_array = np_array[100:]
        np_array = np_array[::8, ::8, 0].flatten()
        np_array[np_array == 255] = 1
        np_array[np_array != 1] = 0
        return np_array

def relu(vector):
    return np.maximum(vector, 0)
