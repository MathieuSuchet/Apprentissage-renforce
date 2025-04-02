import time

from morpion import TicTacToeEnv
import pygame

if __name__ == "__main__":
    # Boucle de jeu
    env = TicTacToeEnv()
    obs, _ = env.reset()

    done = False
    while not done:

        truncated, terminated = False, False
        obs, _ = env.reset()

        while not terminated and not truncated:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    env.close()
                # elif event.type == pygame.MOUSEBUTTONDOWN:
                #     x, y = event.pos
                #     action = (y // env.SQUARE_SIZE) * 3 + (x // env.SQUARE_SIZE)

            action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)