import numpy as np
import os.path

class AI():
    def __init__(self,w1name,w2name):
        self.weights_one_name = w1name
        self.weights_two_name = w2name

        self.weights_one = []
        self.weights_two = []
        self.hidden_layer_size = 500
        self.input_layer_size = 4400

        if os.path.exists(self.weights_one_name) and os.path.exists(self.weights_two_name):
            self.load_weights()

    def load_weights(self):
        self.weights_one = np.loadtxt(self.weights_one_name)
        self.weights_two = np.loadtxt(self.weights_two_name)

    def getAction(self,rgb, paddleA, paddleB, ball, reward, done):
        paddle_y = paddleA.y
        ball_y = ball.y

        return -5 if paddle_y < ball_y else 5 if paddle_y > ball_y else 0