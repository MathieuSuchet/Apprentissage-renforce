from morpion import TicTacToeEnv
from agent import Learner
from observation import Observation
import numpy as np

# 🔧 Paramètres
nb_episodes = 20000
learning_rate = 0.1
discount_factor = 0.95
initial_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.001

# Initialisation
env = TicTacToeEnv()
agent = Learner(lr=learning_rate, gamma=discount_factor, epsilon=initial_epsilon)

# Entraînement
for episode in range(nb_episodes):
    obs, _ = env.reset()
    done = False
    prev_obs = None
    prev_action = None

    # Actualiser epsilon
    agent.epsilon = max(min_epsilon, initial_epsilon * np.exp(-decay_rate * episode))

    while not done:
        action = agent.pick_action(obs)

        next_obs, reward, done, _, _ = env.step(action)

        # Enregistrer transition (agent joue les 2 joueurs)
        if prev_obs is not None:
            # Le joueur précédent reçoit -reward si l'autre a gagné
            if done:
                if reward == 10:  # L'autre a gagné, moi j'ai perdu
                    agent.update_table(prev_obs, prev_action, -10)
                elif reward == 1:  # Match nul
                    agent.update_table(prev_obs, prev_action, 1)
            else:
                agent.update_table(prev_obs, prev_action, 0)


        # Apprendre depuis l’action actuelle
        agent.update_table(obs, action, reward)

        prev_obs = obs
        prev_action = action
        obs = next_obs

    if episode % 1000 == 0:
        print(f"Épisode {episode}/{nb_episodes} | ε = {agent.epsilon:.3f}")

# Sauvegarde du modèle
agent.save("q_table.pkl")
print("✅ Entraînement terminé, Q-table sauvegardée.")
env.close()
