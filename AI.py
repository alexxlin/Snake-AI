from Environment import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

class SnakeAI:
    def __init__(self, train_games = 100, test_games = 1, goal_steps = 2000, lr = 1e-2, filename = 'snake_NN.tflearn'):
        self.train_games = train_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]
        self.left_direction = -1
        self.right_direction = 1

    def train(self):
        training_data = []
        for iter in range(self.train_games):
            game = SnakeGame()
            iter, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for step in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, food  = game.step(game_action)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    #AI made desiable move
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
        return training_data

    def train(self):

        training_data = []

        for iteration in range(self.train_games):
            game = SnakeGame()
            ite, previous_score, snake, food = game.start()
            food_distance = self.get_food_distance(self, snake, food)
            previous_observation = self.generate_action(snake)

            for step in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, food  = game.step(game_action)

                #if the game is done
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break

                else:
                    food_distance = self.get_food_distance(snake, food)
                    #AI made the correct move, increased score or decrease distance to food
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])

                    #AI made the wrong move
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance


    def generate_action(self, snake):
        action = randint(0,2) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction(snake)
        new_direction = snake_direction

        if action == left_direction:
            new_direction = self.turn_left(snake_direction)

        elif action == right_direction:
            new_direction = self.turn_right(snake_direction)

        for pair in self.vectors_and_keys:
            if  new_direction.tolist() == pair[0]:
                game_action = pair[1]

        return game_action

    def generate_observation(self, snake, food):

        snake_direction = self.get_snake_direction(snake)
        food_direction = self.get_food_direction(snake, food)
        blocked_left = self.is_direction_blocked(snake, self.turn_left(snake_direction))
        blocked_front = self.is_direction_blocked(snake, snake_direction)
        blocked_right = self.is_direction_blocked(snake, self.turn_right(snake_direction))
        angle = self.compute_angle(snake_direction, food_direction)

        return np.array([int(blocked_left), int(blocked_front), int(blocked_right), angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def compute_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        activations = ['relu', 'tanh', 'softmax']
        regularizers = ['L2', 'L1']
        network = fully_connected(network, 20, activation = activations[1], regularizer = regularizers[0])
        #network = fully_connected(network, 1, activation='linear')
        
        optimizers = ['rmsprop', 'adam', 'momentum', 'ftrl']
        losses = ['mean_square', 'cross_entropy']
        network = regression(network, optimizer = optimizers[3], learning_rate=self.lr, loss= losses[1], name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 5, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food  = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print(steps)
                    print(snake)
                    print(food)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):
        game = SnakeGame(gui = True)
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)

    def train(self):
        training_data = self.train()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)
        print("done testing")

if __name__ == "__main__":
    SnakeAI().visualise()
  #  SnakeAI().train()