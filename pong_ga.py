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
        self.average = .5
        self.ave_count = 1

    def getAction(self, rgb, paddleA, paddleB, ball, reward, done):
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

    def mutate_weights(self):
        percent = 0.01  # Percent of weights to mutate. NOTE: Possibly decrease (.01 or .001)
        temp_w1 = self.weights_one.flatten()
        temp_w2 = self.weights_two.flatten()

        rand_ind1 = np.random.choice(temp_w1.size, size=int(temp_w1.size * percent))
        rand_ind2 = np.random.choice(temp_w2.size, size=int(temp_w2.size * percent))

        temp_w1[rand_ind1] += np.random.normal(0, 0.1, rand_ind1.size)
        temp_w2[rand_ind2] += np.random.normal(0, 0.1, rand_ind2.size)

        self.weights_one = temp_w1.reshape((hidden_layer_size,input_layer_size))
        self.weights_two = temp_w2.reshape((1, hidden_layer_size))

    def copy(self):
        temp = PongPlayer(self.weights_one.copy(), self.weights_two.copy())  # .copy() here is important
        return temp

@np.vectorize
def sigmoid(x):
    return 1.0 / (1 + np.exp(-1 * x))


def relu(vector):
    return np.maximum(vector, 0)

number_players = 5
tester = training_ai.Tester()

def runGA(weights_one, weights_two):
    number_generations = 1
    pong_players = [0] * number_players

    for p in range(number_players):
        pong_players[p] = PongPlayer(weights_one, weights_two)

    for g in range(number_generations):
        pong_players = runGeneration(pong_players)

    best_player = pong_players[0]
    '''for p in pong_players:
        if best_player.score < p.score:
            best_player = p
    print("Best score of last generation: " + str(best_player.score))'''
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

for i in range(5):
    print("\nrunning generation", i+1, "\n")
    w1 = np.loadtxt("weights_one.txt", delimiter=',')
    # w1 = np.random.randint(-10,10,size=(hidden_layer_size,input_layer_size))
    w2 = np.loadtxt("weights_two.txt", delimiter=',')
    # w2 = np.random.randint(-10,10,size=(1,hidden_layer_size))
    final_player = runGA(w1, w2)
    np.savetxt("weights_one.txt", final_player.weights_one, delimiter=',')
    np.savetxt("weights_two.txt", final_player.weights_two, delimiter=',')
