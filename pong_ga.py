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
        hidden = sigmoid(hidden_unactivated)

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

    def mutate_weights(self):
        percent = 0.01
        temp_w1 = self.weights_one.flatten()
        temp_w2 = self.weights_two.flatten()

        rand_ind1 = np.random.choice(temp_w1.size, size=int(temp_w1.size * percent))
        rand_ind2 = np.random.choice(temp_w2.size, size=int(temp_w2.size * percent))

        temp_w1[rand_ind1] += np.random.normal(0, 0.1, rand_ind1.size)
        temp_w2[rand_ind2] += np.random.normal(0, 0.1, rand_ind2.size)

        self.weights_one = temp_w1.reshape((hidden_layer_size, input_layer_size))
        self.weights_two = temp_w2.reshape((1, hidden_layer_size))

    def copy(self):
        temp = PongPlayer(self.weights_one.copy(), self.weights_two.copy())  # .copy() here is important
        return temp

@np.vectorize
def sigmoid(x):
    return 1.0 / (1 + np.exp(-1 * x))

def crossover(player_one, player_two):
    w1_one_old = player_one.weights_one
    w1_two_old = player_two.weights_one
    w2_one_old = player_one.weights_two
    w2_two_old = player_two.weights_two

    w1_new_preshape = np.concatenate((w1_one_old[:hidden_layer_size/2], w1_two_old[hidden_layer_size/2:]))
    w2_new_preshape = np.concatenate((w2_one_old[0][:hidden_layer_size/2], w2_two_old[0][hidden_layer_size / 2:]))

    w1_new = w1_new_preshape.reshape((hidden_layer_size, input_layer_size))
    w2_new = w2_new_preshape.reshape((1, hidden_layer_size))

    return PongPlayer(w1_new, w2_new)

def relu(vector):
    return np.maximum(vector, 0)

number_players = 5
number_generations = 5
number_eras = 5
tester = training_ai.Tester()

def runGA(weights_one, weights_two):
    pong_players = [0] * number_players

    for p in range(number_players):
        pong_players[p] = PongPlayer(weights_one, weights_two)

    for g in range(number_generations):
        print("\nstarting generation", g+1, "\n")
        pong_players = runGeneration(pong_players)

    best_player = pong_players[0]
    return best_player


def runGeneration(players):
    new_players = []

    max_score = 0
    for (count, p) in enumerate(players, 1):
        game = comp_exec.Game(p, tester, False)
        print("playing game", count)
        game.runComp()
        if max_score < p.score:
            max_score = p.score
    print("max score: " + str(max_score))

    sorted_players = sorted(players, key=lambda x: x.score, reverse=True)

    new_players.append(crossover(sorted_players[0].copy(), sorted_players[1].copy()))
    new_players.append(crossover(sorted_players[1], sorted_players[2]))

    listpos = 2

    while len(new_players) < len(sorted_players):
        new_players.append(sorted_players[listpos].copy().mutate_weights())
        listpos += 1

    return new_players

'''for e in range(number_generations):
    print("\nrunning era", e+1, "\n")
    w1 = np.loadtxt("weights_one.txt", delimiter=',')
    # w1 = np.random.randint(-10,10,size=(hidden_layer_size,input_layer_size))
    w2 = np.loadtxt("weights_two.txt", delimiter=',')
    # w2 = np.random.randint(-10,10,size=(1,hidden_layer_size))
    final_player = runGA(w1, w2)
    np.savetxt("weights_one.txt", final_player.weights_one, delimiter=',')
    np.savetxt("weights_two.txt", final_player.weights_two, delimiter=',')'''

# w1 = np.random.randint(-10,10,size=(hidden_layer_size,input_layer_size))
# w2 = np.random.randint(-10,10,size=(1,hidden_layer_size))
# np.savetxt("weights_one.txt", w1, delimiter=',')
# np.savetxt("weights_two.txt", w2, delimiter=',')
