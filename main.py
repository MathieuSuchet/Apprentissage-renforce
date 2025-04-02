import random
import time

from morpion import TicTacToeEnv
import pygame
from agent import Learner, get_available_actions
from tqdm import tqdm

if __name__ == "__main__":
    # Boucle de jeu
    env = TicTacToeEnv()
    obs, _ = env.reset()
    agent = Learner(
        lr=0.1,
        gamma=0.9,
        epsilon=0
    )

    render = True

    if render:
        agent.load("./models/table.pkl")

    done = False

    if render:
        while not done:

            truncated, terminated = False, False
            obs, _ = env.reset()

            ai_agent = random.randint(1, 2)
            player = 3 - ai_agent

            while not terminated and not truncated:
                action = None

                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        env.close()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos

                        if env.current_player == player:
                            action = (y // env.SQUARE_SIZE) * 3 + (x // env.SQUARE_SIZE)
                            if action not in get_available_actions(obs):
                                action = None

                if env.current_player == ai_agent:
                    action = agent.pick_action(obs)

                if action is not None:
                    obs, reward, terminated, truncated, _ = env.step(action)

                time.sleep(0.2)

    else:
        progress_bar = tqdm(desc="Progression")

        n_steps = 0
        target_steps = 100_000
        while not done:
            truncated, terminated = False, False
            obs, _ = env.reset()

            avg_r = 0
            count = 0

            while not terminated and not truncated:
                action = agent.pick_action(obs)

                if action is not None:
                    n_obs, reward, terminated, truncated, _ = env.step(action)
                    agent.update_table(obs, action, reward)

                    avg_r += reward
                    count += 1

                    obs = n_obs

                n_steps += 1
                done = n_steps >= target_steps
                progress_bar.update(1)



    agent.save("./models/table.pkl")