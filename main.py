"""
Agent Q-Learning pour la Détection et Réponse aux Cyberattaques
Basé sur le paper: "Q-Learning Approach Applied to Network Security"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PARTIE 1 : CONFIGURATION DE L'ENVIRONNEMENT
# ============================================

class CyberSecurityEnvironment:
    """
    Environnement simulé pour l'entraînement de l'agent Q-learning
    """
    def __init__(self):
        # Définition des états
        self.states = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        self.n_states = len(self.states)
        
        # Définition des actions
        self.actions = ['Allow', 'Report', 'Return', 'Block']
        self.n_actions = len(self.actions)
        
        # Matrice de récompenses (états × actions)
        self.rewards = np.array([
            [5, 0, 1, -1],    # Normal: Allow=5, Report=0, Return=1, Block=-1
            [-5, 3, 0, 10],   # DoS: Allow=-5, Report=3, Return=0, Block=10
            [0, 1, 2, 5],     # Probe: Allow=0, Report=1, Return=2, Block=5
            [-2, 1, 4, 6],    # R2L: Allow=-2, Report=1, Return=4, Block=6
            [-3, 0, 3, 8]     # U2R: Allow=-3, Report=0, Return=3, Block=8
        ])
        
        # Matrice de transition (sera estimée ou simulée)
        self.transition_probs = self._initialize_transition_probabilities()
        
    def _initialize_transition_probabilities(self):
        """
        Initialise les probabilités de transition pour chaque action
        Format: {action: matrix (5×5)} où matrix[i,j] = P(j|i,action)
        """
        transitions = {}
        
        # Action 0: Allow (probabilités si on autorise le trafic)
        transitions[0] = np.array([
            [0.90, 0.03, 0.03, 0.02, 0.02],  # Normal → ...
            [0.10, 0.70, 0.10, 0.05, 0.05],  # DoS → ...
            [0.60, 0.10, 0.20, 0.05, 0.05],  # Probe → ...
            [0.50, 0.10, 0.10, 0.20, 0.10],  # R2L → ...
            [0.40, 0.10, 0.10, 0.10, 0.30]   # U2R → ...
        ])
        
        # Action 1: Report (signaler à l'admin)
        transitions[1] = np.array([
            [0.85, 0.05, 0.05, 0.03, 0.02],
            [0.40, 0.40, 0.10, 0.05, 0.05],
            [0.50, 0.10, 0.25, 0.10, 0.05],
            [0.45, 0.10, 0.10, 0.25, 0.10],
            [0.40, 0.10, 0.10, 0.10, 0.30]
        ])
        
        # Action 2: Return (renvoyer pour reclassification)
        transitions[2] = np.array([
            [0.80, 0.05, 0.05, 0.05, 0.05],
            [0.30, 0.50, 0.10, 0.05, 0.05],
            [0.40, 0.10, 0.35, 0.10, 0.05],
            [0.35, 0.10, 0.10, 0.35, 0.10],
            [0.30, 0.10, 0.10, 0.10, 0.40]
        ])
        
        # Action 3: Block (bloquer le trafic)
        transitions[3] = np.array([
            [0.70, 0.10, 0.10, 0.05, 0.05],  # Si on bloque Normal → risque de rester bloqué
            [0.85, 0.08, 0.03, 0.02, 0.02],  # Si on bloque DoS → retour à Normal
            [0.80, 0.05, 0.10, 0.03, 0.02],  # Si on bloque Probe → retour à Normal
            [0.80, 0.05, 0.05, 0.07, 0.03],  # Si on bloque R2L → retour à Normal
            [0.85, 0.05, 0.03, 0.02, 0.05]   # Si on bloque U2R → retour à Normal
        ])
        
        return transitions
    
    def reset(self):
        """Réinitialise l'environnement à un état aléatoire"""
        return random.randint(0, self.n_states - 1)
    
    def step(self, state, action):
        """
        Exécute une action dans un état donné
        Retourne: (next_state, reward, done)
        """
        # Obtenir la récompense
        reward = self.rewards[state, action]
        
        # Transition vers le prochain état basée sur les probabilités
        transition_prob = self.transition_probs[action][state]
        next_state = np.random.choice(self.n_states, p=transition_prob)
        
        # Episode terminé aléatoirement (10% de chance)
        done = random.random() < 0.1
        
        return next_state, reward, done


# ============================================
# PARTIE 2 : AGENT Q-LEARNING
# ============================================

class QLearningAgent:
    """
    Agent Q-Learning pour la réponse aux intrusions
    """
    def __init__(self, n_states, n_actions, learning_rate=0.2, 
                 discount_factor=0.1, epsilon=0.9, epsilon_decay=0.05, 
                 epsilon_min=0.01):
        
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Hyperparamètres
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table (actions × états)
        self.Q = np.zeros((n_actions, n_states))
        
        # Historique
        self.episode_rewards = []
        
    def select_action(self, state):
        """
        Stratégie ε-greedy pour sélectionner une action
        """
        if random.random() < self.epsilon:
            # Exploration: action aléatoire
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: meilleure action connue
            return np.argmax(self.Q[:, state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Mise à jour de la Q-table selon l'équation de Bellman
        Q'(s,a) = (1-α)Q(s,a) + α[r + γ·max(Q(s',a'))]
        """
        current_q = self.Q[action, state]
        max_future_q = np.max(self.Q[:, next_state])
        
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        
        self.Q[action, state] = new_q
    
    def decay_epsilon(self):
        """Réduction progressive de l'exploration"""
        self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), self.epsilon_min)
    
    def get_optimal_policy(self, states):
        """
        Extrait la politique optimale de la Q-table
        """
        policy = {}
        actions = ['Allow', 'Report', 'Return', 'Block']
        
        for state_idx, state_name in enumerate(states):
            best_action_idx = np.argmax(self.Q[:, state_idx])
            policy[state_name] = actions[best_action_idx]
        
        return policy


# ============================================
# PARTIE 3 : ENTRAÎNEMENT
# ============================================

def train_agent(env, agent, num_episodes=300, steps_per_episode=50):
    """
    Entraîne l'agent Q-learning dans l'environnement simulé
    """
    print("="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*60)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(steps_per_episode):
            # Sélectionner une action
            action = agent.select_action(state)
            
            # Exécuter l'action
            next_state, reward, done = env.step(state, action)
            
            # Mettre à jour la Q-table
            agent.update_q_value(state, action, reward, next_state)
            
            # Accumuler la récompense
            total_reward += reward
            
            # Transition
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Enregistrer la récompense
        agent.episode_rewards.append(total_reward)
        
        # Afficher progression
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(agent.episode_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\n✓ Entraînement terminé !")
    return agent


# ============================================
# PARTIE 4 : VISUALISATION DES RÉSULTATS
# ============================================

def plot_training_results(agent, window=30):
    """
    Visualise la progression de l'entraînement
    """
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Récompenses par épisode
    plt.subplot(1, 2, 1)
    plt.plot(agent.episode_rewards, alpha=0.6, label='Episode Reward')
    
    # Moyenne mobile
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, 
                                np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(agent.episode_rewards)), 
                moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 2: Q-table heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(agent.Q, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
                yticklabels=['Allow', 'Report', 'Return', 'Block'],
                cbar_kws={'label': 'Q-Value'})
    plt.title('Final Q-Table', fontsize=14, fontweight='bold')
    plt.xlabel('States', fontsize=12)
    plt.ylabel('Actions', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Graphique sauvegardé: training_results.png")


def display_q_table(agent, env):
    """
    Affiche la Q-table sous forme de DataFrame
    """
    df = pd.DataFrame(
        agent.Q,
        columns=env.states,
        index=env.actions
    )
    print("\n" + "="*60)
    print("Q-TABLE FINALE")
    print("="*60)
    print(df.to_string())
    print()


def display_optimal_policy(policy):
    """
    Affiche la politique optimale
    """
    print("\n" + "="*60)
    print("POLITIQUE OPTIMALE")
    print("="*60)
    for state, action in policy.items():
        emoji = "✅" if action == "Allow" else "🛡️"
        print(f"{emoji} {state:10s} → {action}")
    print()


# ============================================
# PARTIE 5 : TEST SUR NSL-KDD
# ============================================

class NSLKDDTester:
    """
    Classe pour tester l'agent sur le dataset NSL-KDD
    """
    def __init__(self):
        # Mapping des attaques NSL-KDD vers nos 5 états
        self.attack_mapping = {
            'normal': 0,
            # DoS attacks
            'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 
            'teardrop': 1, 'mailbomb': 1, 'apache2': 1, 'processtable': 1, 
            'udpstorm': 1,
            # Probe attacks
            'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 
            'saint': 2,
            # R2L attacks
            'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 
            'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3, 
            'sendmail': 3, 'named': 3, 'snmpgetattack': 3, 'snmpguess': 3, 
            'xlock': 3, 'xsnoop': 3, 'worm': 3,
            # U2R attacks
            'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 
            'httptunnel': 4, 'ps': 4, 'sqlattack': 4, 'xterm': 4
        }
    
    def load_nsl_kdd(self, filepath='KDDTest+.txt', sample_size=None):
        """
        Charge et prétraite le dataset NSL-KDD
        
        Note: Si vous n'avez pas le fichier, ce code génère des données synthétiques
        """
        try:
            # Essayer de charger le vrai dataset
            column_names = self._get_column_names()
            data = pd.read_csv(filepath, names=column_names)
            print(f"✓ Dataset NSL-KDD chargé: {len(data)} échantillons")
            
        except FileNotFoundError:
            # Générer des données synthétiques
            print("⚠️  Fichier NSL-KDD non trouvé. Génération de données synthétiques...")
            data = self._generate_synthetic_data(sample_size or 5000)
        
        # Échantillonner si demandé
        if sample_size and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
            print(f"✓ Échantillonnage: {len(data)} échantillons")
        
        return data
    
    def _get_column_names(self):
        """Retourne les noms de colonnes du dataset NSL-KDD"""
        return ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                'num_failed_logins', 'logged_in', 'num_compromised', 
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                'num_shells', 'num_access_files', 'num_outbound_cmds', 
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 
                'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
                'attack', 'difficulty']
    
    def _generate_synthetic_data(self, n_samples=5000):
        """
        Génère des données synthétiques simulant NSL-KDD
        """
        np.random.seed(42)
        
        # Distribution des attaques (similaire à NSL-KDD)
        attack_types = ['normal'] * 2500 + \
                      ['neptune', 'smurf', 'back'] * 800 + \
                      ['portsweep', 'ipsweep'] * 300 + \
                      ['guess_passwd', 'warezmaster'] * 150 + \
                      ['buffer_overflow', 'rootkit'] * 50
        
        random.shuffle(attack_types)
        attack_types = attack_types[:n_samples]
        
        # Features numériques simulées
        data = {
            'duration': np.random.exponential(2, n_samples),
            'src_bytes': np.random.exponential(500, n_samples),
            'dst_bytes': np.random.exponential(300, n_samples),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(8, n_samples),
            'serror_rate': np.random.uniform(0, 1, n_samples),
            'srv_serror_rate': np.random.uniform(0, 1, n_samples),
            'rerror_rate': np.random.uniform(0, 1, n_samples),
            'same_srv_rate': np.random.uniform(0, 1, n_samples),
            'diff_srv_rate': np.random.uniform(0, 1, n_samples),
            'attack': attack_types
        }
        
        df = pd.DataFrame(data)
        print(f"✓ Données synthétiques générées: {len(df)} échantillons")
        
        return df
    
    def preprocess_data(self, data):
        """
        Prétraitement: sélection de features et mapping des états
        """
        # Sélectionner les features numériques importantes
        numeric_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 
                           'srv_count', 'serror_rate', 'srv_serror_rate', 
                           'rerror_rate', 'same_srv_rate', 'diff_srv_rate']
        
        # Garder seulement les features disponibles
        available_features = [f for f in numeric_features if f in data.columns]
        X = data[available_features].fillna(0)
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Mapper les attaques vers les 5 états
        y = data['attack'].apply(lambda x: self.attack_mapping.get(x, 0))
        
        return X_scaled, y
    
    def test_agent(self, agent, X, y, env):
        """
        Teste l'agent sur les données NSL-KDD
        """
        print("\n" + "="*60)
        print("TEST SUR NSL-KDD")
        print("="*60)
        
        cumulative_reward = 0
        actions_taken = []
        expected_actions = []
        
        for i in range(len(y)):
            state = y.iloc[i]
            
            # L'agent choisit la meilleure action
            action = np.argmax(agent.Q[:, state])
            actions_taken.append(action)
            
            # Action attendue (ground truth)
            if state == 0:  # Normal
                expected_actions.append(0)  # Allow
            else:  # Attaque
                expected_actions.append(3)  # Block
            
            # Récompense
            reward = env.rewards[state, action]
            cumulative_reward += reward
        
        # Calcul des métriques
        accuracy = accuracy_score(expected_actions, actions_taken)
        
        print(f"\n📊 RÉSULTATS:")
        print(f"  • Échantillons testés: {len(y)}")
        print(f"  • Récompense cumulative: {cumulative_reward:.2f}")
        print(f"  • Récompense moyenne: {cumulative_reward/len(y):.2f}")
        print(f"  • Accuracy: {accuracy:.2%}")
        
        # Distribution des états
        state_counts = y.value_counts().sort_index()
        print(f"\n📈 DISTRIBUTION DES ÉTATS:")
        state_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        for idx, count in state_counts.items():
            print(f"  • {state_names[idx]:10s}: {count:5d} ({count/len(y)*100:.1f}%)")
        
        # Matrice de confusion
        self._plot_confusion_matrix(expected_actions, actions_taken)
        
        return {
            'cumulative_reward': cumulative_reward,
            'accuracy': accuracy,
            'actions_taken': actions_taken,
            'expected_actions': expected_actions
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Affiche la matrice de confusion
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Allow', 'Report', 'Return', 'Block'],
                   yticklabels=['Allow', 'Report', 'Return', 'Block'])
        plt.title('Confusion Matrix: Expected vs Actual Actions', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Action', fontsize=12)
        plt.ylabel('Expected Action', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Matrice de confusion sauvegardée: confusion_matrix.png")


# ============================================
# PARTIE 6 : FONCTION PRINCIPALE
# ============================================

def main():
    """
    Fonction principale pour exécuter tout le pipeline
    """
    print("\n" + "="*60)
    print("AGENT Q-LEARNING POUR DÉTECTION DE CYBERATTAQUES")
    print("="*60 + "\n")
    
    # 1. Créer l'environnement
    print("📋 ÉTAPE 1: Création de l'environnement")
    env = CyberSecurityEnvironment()
    print(f"  • États: {env.states}")
    print(f"  • Actions: {env.actions}")
    
    # 2. Créer l'agent
    print("\n🤖 ÉTAPE 2: Initialisation de l'agent")
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.2,
        discount_factor=0.1,
        epsilon=0.9,
        epsilon_decay=0.05,
        epsilon_min=0.01
    )
    print(f"  • Learning rate (α): {agent.alpha}")
    print(f"  • Discount factor (γ): {agent.gamma}")
    print(f"  • Exploration rate (ε): {agent.epsilon}")
    
    # 3. Entraîner l'agent
    print("\n🎯 ÉTAPE 3: Entraînement")
    agent = train_agent(env, agent, num_episodes=300, steps_per_episode=50)
    
    # 4. Afficher les résultats
    print("\n📊 ÉTAPE 4: Analyse des résultats")
    display_q_table(agent, env)
    
    policy = agent.get_optimal_policy(env.states)
    display_optimal_policy(policy)
    
    avg_reward = np.mean(agent.episode_rewards[-50:])
    print(f"💰 Récompense moyenne (50 derniers épisodes): {avg_reward:.2f}")
    
    # 5. Visualiser
    print("\n📈 ÉTAPE 5: Visualisation")
    plot_training_results(agent, window=30)
    
    # 6. Tester sur NSL-KDD
    print("\n🧪 ÉTAPE 6: Test sur NSL-KDD")
    tester = NSLKDDTester()
    
    # Charger les données (synthétiques si fichier absent)
    data = tester.load_nsl_kdd(sample_size=5000)
    
    # Prétraiter
    X, y = tester.preprocess_data(data)
    
    # Tester
    results = tester.test_agent(agent, X, y, env)
    
    print("\n" + "="*60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("="*60)
    
    return agent, env, results


# ============================================
# EXÉCUTION
# ============================================

if __name__ == "__main__":
    agent, env, results = main()