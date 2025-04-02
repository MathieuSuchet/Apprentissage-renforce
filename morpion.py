import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from observation import Observation

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=2, shape=(10, ), dtype=int)
        
        # Initialisation de Pygame
        pygame.init()
        self.WIDTH, self.HEIGHT = 300, 300
        self.SQUARE_SIZE = self.WIDTH // 3
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Morpion Gymnasium')
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        self.LINE_COLOR = (0, 0, 0)
        self.screen.fill(self.YELLOW)

    def reset(self, seed=None, options=None):
        self.board.fill(0)
        self.current_player = random.randint(1, 2)
        return Observation(self.current_player, self.board.flatten()), {}

    def step(self, action):
        row, col = divmod(action, 3)

        self.board[row, col] = self.current_player
        if self.check_win(self.current_player) == 3:
            return Observation(self.current_player, self.board.flatten()), 10, True, False, {}
        if np.all(self.board != 0):
            return Observation(self.current_player, self.board.flatten()), 0, True, False, {}

        reward = self.check_win(self.current_player)
        cnt_player = self.current_player
        self.current_player = 3 - self.current_player

        return Observation(cnt_player, self.board.flatten()), reward * 2, False, False, {}

    def check_win(self, player):
        max_n = 0
        for row in range(3):
            max_n = max(np.sum(self.board[row, :] == player), max_n)
        for col in range(3):
            max_n = max(np.sum(self.board[:, col] == player), max_n)
        max_n = max(np.sum(np.diag(self.board) == player), max_n)
        max_n = max(np.sum(np.diag(np.fliplr(self.board)) == player), max_n)
        return max_n

    def render(self):
        self.screen.fill(self.YELLOW)
        for i in range(1, 3):
            pygame.draw.line(self.screen, self.LINE_COLOR, (0, i * self.SQUARE_SIZE), (self.WIDTH, i * self.SQUARE_SIZE), 5)
            pygame.draw.line(self.screen, self.LINE_COLOR, (i * self.SQUARE_SIZE, 0), (i * self.SQUARE_SIZE, self.HEIGHT), 5)
        
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 1:
                    pygame.draw.circle(self.screen, self.PURPLE, (col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2, row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2), self.SQUARE_SIZE // 3, 5)
                elif self.board[row, col] == 2:
                    pygame.draw.line(self.screen, self.PURPLE, (col * self.SQUARE_SIZE + 20, row * self.SQUARE_SIZE + 20), (col * self.SQUARE_SIZE + self.SQUARE_SIZE - 20, row * self.SQUARE_SIZE + self.SQUARE_SIZE - 20), 5)
                    pygame.draw.line(self.screen, self.PURPLE, (col * self.SQUARE_SIZE + 20, row * self.SQUARE_SIZE + self.SQUARE_SIZE - 20), (col * self.SQUARE_SIZE + self.SQUARE_SIZE - 20, row * self.SQUARE_SIZE + 20), 5)
        
        pygame.display.flip()

    def close(self):
        pygame.quit()
