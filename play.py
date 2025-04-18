from morpion import TicTacToeEnv
from agent import Learner
import random
import time

# Agent aléatoire
def random_agent(obs):
    return random.choice([i for i, val in enumerate(obs.board) if val == 0])

# Chargement
env = TicTacToeEnv()
agent = Learner(lr=0, gamma=0.95, epsilon=0)  # pas d'exploration
agent.load("q_table.pkl")

# Stats
win, draw, lose = 0, 0, 0

for game in range(5):
    print(f"\n🎮 Partie {game + 1}")
    obs, _ = env.reset()
    done = False
    agent_player = 1 if game % 2 == 0 else 2

    while not done:
        env.render()
        time.sleep(0.5)

        if obs.player_sign == agent_player:
            action = agent.pick_action(obs)
        else:
            action = random_agent(obs)

        obs, reward, done, _, _ = env.step(action)

    env.render()

    # Résultat
    if reward == 10 and obs.player_sign == agent_player:
        print("✅ L'agent a gagné !")
        win += 1
    elif reward == 10 and obs.player_sign != agent_player:
        print("❌ L'agent a perdu.")
        lose += 1
    else:
        print("🤝 Match nul.")
        draw += 1

# Résumé
print("\n📊 Résumé après 5 parties :")
print(f"✅ Victoires : {win}")
print(f"🤝 Nuls     : {draw}")
print(f"❌ Défaites : {lose}")

env.close()
