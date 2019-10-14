import gym
import numpy as np

batch_size = 5
hidden_layer_size = 200
input_layer_size = 6480
output_layer_size = 1
num_frames_in_game = 2000  # Number of frames to allow the ai to play for (a full game usually less than 1500)

# Initializations
env = gym.make('Pong-v0')  # Setting up the pong game environment
average = 0

# The main method, controls the flow of the program
def main():
    av_count = 0
    global average
    weights_one = np.random.randn(hidden_layer_size, input_layer_size)
    # print(weights_one)
    weights_two = np.random.randn(output_layer_size, hidden_layer_size)

    for g in range(batch_size):  # Plays batch_size games before updating weights
        print("Resetting environment\n")
        observation = env.reset()  # Reset environment for new game

        for frame in range(num_frames_in_game):  # Continues until # frames have run

            env.render()  # Display the pong screen

            proc_input = processScreen(observation)  # Process the screen and save
            # Note that the very first frame of the game is a different color than all of the rest

            # Get the action and probability of moving up using the processed input and weight matrices:
            (action, prob) = getAction(proc_input, weights_one, weights_two)

            observation, reward, done, info = env.step(action)  # Get relevant information, useful gym statistics
            # It should also be mentioned that the "env.step(action)" portion of this line actually indicates
            # what direction to move the paddle in the given frame
            av_count = av_count + 1
            average = (average + prob) / (av_count)

            if done:  # The game has ended
                break  # Break out of game loop

        print("Game " + str(g + 1) + " finished after " + str(frame) + " frames")

    print("Closing environment")
    env.close()


def processScreen(np_array):
    np_array = np_array[33:195]  # Crop out the boxes and score
    np_array = np_array[::2, ::2, 0]  # Reduce resolution and only grab red component

    processed_screen = np_array.flatten()  # Turn into a nx1 matrix

    processed_screen[(processed_screen != 109) & (processed_screen != 144)] = 1  # Sets all non white pixels to 0
    processed_screen[processed_screen != 1] = 0  # Sets all white pixels to 1

    return processed_screen


def getAction(proc_input, weights_one, weights_two):
    # Compute hidden layer
    hidden = relu(np.dot(weights_one, proc_input))
    # Compute output layer
    output = relu(np.dot(weights_two, hidden))
    # print(output)
    prob = output[0]
    # print(prob)

    if prob < average:  # If the probability of moving up is less than 50%
        return (3, prob)  # 3 corresponds to moving down in gym
    else:
        return (2, prob)  # 2 corresponds to moving up in gym


# Sigmoid function: takes in an iterable and returns an iterable with the sigmoid function applied to all values
@np.vectorize
def sigmoid(x):
    return 1.0 / (1 + np.exp(-1 * x))


# Relu function: takes in an iterable and returns an iterable with relu function applied to all values
def relu(vector):  # Don't know why you need this, but it's better
    vector[vector < 0] = 0
    return vector


if __name__ == "__main__":
    main()

