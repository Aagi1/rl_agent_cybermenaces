# PPO-based Cyber Defense Agent using CybORG 🛡️ 

Ce projet implémente un agent de défense cyber utilisant **l’apprentissage par renforcement profond (Deep Reinforcement Learning)**. L’agent, entraîné avec l’algorithme **Proximal Policy Optimization (PPO)**, apprend à protéger un réseau simulé contre un attaquant automatisé dans l’environnement **CybORG**.

<img width="500" height="303" alt="image" src="https://github.com/user-attachments/assets/0900f000-d645-4e7f-b885-a14d9e4d12af" />


## Objectif du projet

L’objectif principal est d’étudier la capacité d’un agent intelligent à :
- prendre des décisions défensives adaptées dans un environnement cyber dynamique,
- limiter les dégâts causés par un attaquant,
- améliorer ses performances au fil de l’entraînement, sans supervision explicite.

L’agent défensif (*Blue*) est entraîné contre un attaquant préprogrammé (*Red – B_lineAgent*).



## Technologies utilisées

- **Python**
- **PyTorch** – implémentation du réseau Actor-Critic
- **CybORG** – environnement de simulation cybersécurité
- **Proximal Policy Optimization (PPO)**
- **NumPy**


## Architecture générale

Le système repose sur une architecture **Actor-Critic** :

- **Actor** : apprend une politique de défense (choix des actions)
- **Critic** : estime la valeur des états
- **Environnement** : CybORG (scénario `Scenario1b.yaml`)
- **Adversaire** : agent Red (`B_lineAgent`)

L’agent Blue interagit avec l’environnement, collecte des trajectoires, calcule les avantages (GAE) et met à jour sa politique via PPO.

## Entraînement

L’entraînement se fait sur plusieurs épisodes, chaque épisode correspondant à une simulation complète de défense du réseau.

Principaux paramètres :
- Nombre d’épisodes : 2000
- Steps maximum par épisode : 30
- Actions valides limitées pour éviter des pénalités inutiles
- Utilisation de **Generalized Advantage Estimation (GAE)**

Pour lancer l’entraînement :

```bash
python main.py 
```

Les modèles sont automatiquement sauvegardés :

best_ppo_model.pth : meilleure performance observée

final_ppo_model.pth : état final du modèle
