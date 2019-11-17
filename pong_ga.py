import numpy as np
import comp_exec
import training_ai
import random
import time

input_layer_size = 4400  # Size of processed screen matrix
hidden_layer_size = 500  # Hidden layer size

class PongPlayer:

    def __init__(self, weights_one, weights_two):
        self.weights_one = weights_one
        self.weights_two = weights_two
        self.score = 0

    def getAction(self, rgb, paddleA, paddleB, ball, reward, done):
        input_layer = self.screen_process(rgb)

        hidden_unactivated = np.dot(self.weights_one, input_layer)
        hidden = relu(hidden_unactivated)

        output_unactivated = np.dot(self.weights_two, hidden)
        output = sigmoid(output_unactivated)

        self.score += reward

        if output <= .5:
            return 5
        else:
            return -5

    def screen_process(self, np_array):
        np_array = np_array[100:]
        np_array = np_array[::8, ::8, 0].flatten()
        np_array[np_array == 255] = 1
        np_array[np_array != 1] = 0
        return np_array

    def mutate_weights(self):  # NOTE: Possibly consider using map() instead, would be WAY simpler and possibly faster
        percent = 0.05  # Percent of weights to mutate. NOTE: Possibly decrease (.01 or .001)
        temp_w1 = self.weights_one.flatten()
        temp_w2 = self.weights_two.flatten()

        rand_ind1 = np.random.choice(temp_w1.size, size=int(temp_w1.size * percent))
        rand_ind2 = np.random.choice(temp_w2.size, size=int(temp_w2.size * percent))

        # temp_w1[rand_ind1] = np.random.randn(rand_ind1.size)
        # temp_w2[rand_ind2] = np.random.randn(rand_ind2.size)

        temp_w1[rand_ind1] += np.random.normal(0, 0.1, rand_ind1.size)
        temp_w2[rand_ind2] += np.random.normal(0, 0.1, rand_ind2.size)

        self.weights_one = temp_w1.reshape((hidden_layer_size,input_layer_size))
        self.weights_two = temp_w2.reshape((1, hidden_layer_size))

    def copy(self):
        return PongPlayer(self.weights_one.copy(), self.weights_two.copy())  # .copy() here is important

@np.vectorize
def sigmoid(x):
    return 1.0 / (1 + np.exp(-1 * x))


def relu(vector):
    vector[vector < 0] = 0
    return vector

number_players = 1
tester = training_ai.Tester()

def runGA(weights_one, weights_two):
    number_generations = 1
    pong_players = [0] * number_players

    for p in range(number_players):
        pong_players[p] = PongPlayer(weights_one, weights_two)
    print("Initialized first generation of players.")

    for g in range(number_generations):
        print("\nRunning new generation " + str(g + 1) + "\n")
        pong_players = runGeneration(pong_players)

    best_player = pong_players[0]
    for p in pong_players:
        if best_player.score < p.score:
            best_player = p
    print("Best score of last generation: " + str(best_player.score))
    time.sleep(2)

    return best_player


def runGeneration(players):
    new_players = []

    max_score = 0
    count = 0
    for p in players:
        count = count + 1
        game = comp_exec.Game(p, tester, False)
        print("playing game" + str(count))
        game.runComp()
        if max_score < p.score:
            max_score = p.score
    print("max score: " + str(max_score))

    for p in players:
        if p.score > random.randint(0, max_score):
            temp = p.copy()
            temp.mutate_weights()
            new_players.append(temp)
    while len(new_players) < len(players):
        p = random.choice(players)
        temp = p.copy()
        temp.mutate_weights()
        new_players.append(temp)
    return new_players

w1 = np.loadtxt("weights_one.txt", delimiter=',')
w2 = np.loadtxt("weights_two.txt", delimiter=',')

final_player = runGA(w1, w2)

np.savetxt("weights_one.txt", final_player.weights_one, delimiter=',')
np.savetxt("weights_two.txt", final_player.weights_two, delimiter=',')
