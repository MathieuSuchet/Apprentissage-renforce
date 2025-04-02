import numpy as np

class Observation:
    def __init__(self, player_sign: int, board: np.ndarray):
        self.player_sign = player_sign
        self.board = board