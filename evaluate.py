import matplotlib.pyplot as plt
from morpion import TicTacToeEnv
from agent import Learner
import random

# Joueur aléatoire
def random_agent(obs):
    return random.choice([i for i, val in enumerate(obs.board) if val == 0])

env = TicTacToeEnv()
agent = Learner(lr=0, gamma=0.95, epsilon=0)
agent.load("q_table.pkl")

nb_total_tests = 10  # nombre de points (ex: 10 tests entre 0 et 20k)
games_per_test = 100
step_size = 2000  # tous les 2000 épisodes
episode_steps = []
win_rates = []
draw_rates = []
loss_rates = []

# Simulation sur différents épisodes
for step in range(0, nb_total_tests * step_size + 1, step_size):
    win, draw, lose = 0, 0, 0

    for i in range(games_per_test):
        obs, _ = env.reset()
        done = False
        agent_player = 1 if i % 2 == 0 else 2  # alterne 1er / 2e cad l'agent en premier ou pas 

        while not done:
            if obs.player_sign == agent_player:
                action = agent.pick_action(obs)
            else:
                action = random_agent(obs)

            obs, reward, done, _, _ = env.step(action)

        if reward == 10 and obs.player_sign == agent_player:
            win += 1
        elif reward == 10 and obs.player_sign != agent_player:
            lose += 1
        else:
            draw += 1

    episode_steps.append(step)
    win_rates.append(win / games_per_test * 100)
    draw_rates.append(draw / games_per_test * 100)
    loss_rates.append(lose / games_per_test * 100)

    print(f"Épisodes {step} → Victoires: {win}, Nuls: {draw}, Défaites: {lose}")

# 🎯 Tracer la courbe d’apprentissage
plt.figure(figsize=(10, 6))
plt.plot(episode_steps, win_rates, label="✅ Victoires")
plt.plot(episode_steps, draw_rates, label="🤝 Nuls")
plt.plot(episode_steps, loss_rates, label="❌ Défaites")
plt.xlabel("Épisodes d'entraînement")
plt.ylabel("Pourcentage (%) sur 100 parties")
plt.title("Courbe d'apprentissage de l'agent (évaluation vs joueur aléatoire)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nRésultat final sur {nb_total_tests * games_per_test} parties :")
print(f"✅ Victoires moyennes : {sum(win_rates)/len(win_rates):.2f}%")
print(f"🤝 Nuls moyens        : {sum(draw_rates)/len(draw_rates):.2f}%")
print(f"❌ Défaites moyennes  : {sum(loss_rates)/len(loss_rates):.2f}%")
env.close()


# Fin de la simulation
