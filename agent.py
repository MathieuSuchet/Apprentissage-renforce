import os
import pathlib
import pickle
import random
from typing import List, Dict, Tuple, Union

import numpy as np

from observation import Observation


def get_available_actions(obs: Observation):
    return [i for i, mark in enumerate(env_to_str(obs)[:-1]) if mark == ' ']

def env_to_str(obs: Observation):
    if obs.board.size > 1 >= np.prod(obs.board.shape):
        return ""
    else:
        p_to_s = {
            0: " ",
            1: "O",
            2: "X"
        }

        return "".join(([p_to_s[i] for i in obs.board])) + str(obs.player_sign)


class Learner:
    def __init__(self, lr: float, gamma: float, epsilon: float):
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self._q_table: Dict[Tuple[str, int], float] = {}

    def update_table(self, state, action, reward):
        q_table_val = self._q_table.get((state, action), 0)
        max_q_val = max([self._q_table.get((state, a), 0) for a in get_available_actions(state)], default=0)

        self._q_table[(env_to_str(state), action)] = q_table_val + self.lr * (reward + self.gamma * max_q_val - q_table_val)

    def pick_action(self, obs: Observation):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(get_available_actions(obs))
        else:
            q_values = {a: self._q_table.get((env_to_str(obs), a), 0) for a in get_available_actions(obs)}
            filtered_q_values = {}

            for act, val in q_values.items():
                if val == max(q_values.values()):
                    filtered_q_values[act] = val
            return random.choice(list(filtered_q_values.keys()))

    def _save_table(self, path: pathlib.Path):
        if path.exists() and path.is_file():
            with path.open("wb") as f:
                pickle.dump(self._q_table, f)
        else:
            with path.open("xb") as f:
                pickle.dump(self._q_table, f)

    def _load_table(self, path: pathlib.Path):
        if path.exists() and path.is_file():
            with path.open("rb") as f:
                self._q_table = pickle.load(f)
        else:
            raise FileNotFoundError(f"Impossible d'ouvrir le fichier {path}")


    def save(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        self._save_table(path)

    def load(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        self._load_table(path)
        
