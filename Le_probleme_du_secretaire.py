'''
Projet
le problème du secrétaire

Groupe: Mathis Chabaud, Gerard Kra Kouame, David Man, Seyed Hossein Seyedi

Last up-date
24/06/2024

'''


import numpy as np
import matplotlib.pyplot as plt

#%% Fonctions

def calculer_ut_etoile(N):
    """
    Calcule les valeurs de u_t*(0) et u_t*(1) pour le problème du secrétaire.

    Parameters:
    N (int): Nombre total de candidats

    Returns:
    tuple: Deux tableaux contenant les valeurs de u_t*(0) et u_t*(1)
    """
    # Initialisation des tableaux pour contenir u_t*(0) et u_t*(1)
    u0 = np.zeros(N + 1)
    u1 = np.zeros(N + 1)
    u1[N] = 1  # Dernier candidat, s'il est le meilleur vu jusqu'ici, doit être choisi

    # Induction arrière pour remplir u1 et u0
    for t in reversed(range(1, N)):
        u0[t] = (1 / (t + 1)) * u1[t + 1] + (t / (t + 1)) * u0[t + 1]
        u1[t] = max(t / N, u0[t])

    return u0, u1

def trouver_tau_optimal(N, u1):
    """
    Trouve la valeur optimale de tau pour le problème du secrétaire.

    Parameters:
    N (int): Nombre total de candidats
    u1 (array): Tableau des valeurs de u_t*(1)

    Returns:
    int: La valeur optimale de tau
    """
    if N > 2:
        for tau in reversed(range(1, N)):
            if u1[tau] > tau / N:
                return tau
    else:
        return 1  # N<=2

#%% Graphiques Proportion de Candidats à Interviewer 
#   et Probabilité de Sélectionner le Meilleur Candidat vs N

max_n = 50

tauxs = []  # Taux initiaux
probabilites = []  # Probabilités initiales
ns = range(1, max_n + 1)

for N in ns:
    _, u1 = calculer_ut_etoile(N)
    tau = trouver_tau_optimal(N, u1)
    tauxs.append(tau / N)
    probabilites.append(u1[tau])


plt.figure(figsize=(14, 6))
plt.plot(ns, tauxs, marker='.')
plt.xlabel('Nombre de Candidats (N)')
plt.ylabel('Proportion de Candidats Interviewés (τ/N)')
plt.title('Proportion de Candidats à Interviewer')
plt.axhline(y=np.exp(-1), color='r', linestyle='--', label='1 / e')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(ns, probabilites, marker='.')
plt.xlabel('Nombre de Candidats (N)')
plt.ylabel('Probabilité de Choisir le Meilleur Candidat')
plt.title('Probabilité de Sélectionner le Meilleur Candidat vs N')
plt.axhline(y=np.exp(-1), color='r', linestyle='--', label='1 / e')
plt.legend()

plt.tight_layout()
plt.show()


#%% Graphique évolution u*(0) et u*(1)


N = 1000

u0, u1 = calculer_ut_etoile(N)
ns = range(1, N+1)
tau = int(N * (1/np.exp(1)))

plt.figure(figsize=(14, 6))
plt.plot(ns, u0[1:], label = r'$u^*(0)$')
plt.plot(ns, u1[1:], label = r'$u^*(1)$')

plt.axvline(x=tau, color='r', linestyle='--', label= fr'$\tau = {tau}$')
plt.text(tau, max(u0) * 0.8, fr'$\tau = {tau}$', color='r', ha='left')

plt.xlabel('Périodes t')
plt.ylabel('Probabilité de Choisir le Meilleur Candidat')
plt.title(f'Probabilité vs t [N = {N}]')

plt.legend()
plt.show()



#%% Fonctions simulation strategies


def simulation_strategie_percent(N, percent):
    """
    Simule une stratégie où un certain pourcentage de candidats est observé.

    Parameters:
    N (int): Nombre total de candidats
    percent (float): Pourcentage de candidats à observer

    Returns:
    bool: True si le meilleur candidat est choisi, False sinon
    """
    tau = int(N * percent)
    candidats = np.random.permutation(N) + 1
    meilleur_candidat = N
    meilleur_vu = 0
    
    #Observation candidats

    for t in range(tau):
        if candidats[t] > meilleur_vu:
            meilleur_vu = candidats[t]
            
    #Selection prochain meilleure

    for t in range(tau, N):
        if candidats[t] > meilleur_vu:
            return candidats[t] == meilleur_candidat

    return False

def simulation_strategie_aleatoire(N):
    """
    Simule la stratégie aléatoire pour le problème du secrétaire.

    Parameters:
    N (int): Nombre total de candidats

    Returns:
    bool: True si le meilleur candidat est choisi, False sinon
    """
    candidats = np.random.permutation(N) + 1
    meilleur_candidat = N
    choix = np.random.choice(N)
    return candidats[choix] == meilleur_candidat

def comparer_strategies(N, simulations):
    """
    Compare les différentes stratégies en simulant plusieurs fois le processus de sélection.

    Parameters:
    N (int): Nombre total de candidats
    simulations (int): Nombre de simulations à effectuer

    Returns:
    dict: Dictionnaire contenant les probabilités de succès pour chaque stratégie
    """
    
    tau_optimal = 1 / np.exp(1)

    succes_aleatoire = 0
    succes_50_percent = 0
    succes_80_percent = 0
    succes_20_percent = 0
    succes_67_percent = 0
    succes_optimal = 0

    for _ in range(simulations):
        if simulation_strategie_aleatoire(N):
            succes_aleatoire += 1
        if simulation_strategie_percent(N, 0.50):
            succes_50_percent += 1
        if simulation_strategie_percent(N, 0.80):
            succes_80_percent += 1
        if simulation_strategie_percent(N, 1/5):
            succes_20_percent += 1
        if simulation_strategie_percent(N, 2/3):
            succes_67_percent += 1
        if simulation_strategie_percent(N, tau_optimal):
            succes_optimal += 1

    return {
        "aleatoire": succes_aleatoire / simulations,
        "50_percent": succes_50_percent / simulations,
        "80_percent": succes_80_percent / simulations,
        "20_percent": succes_20_percent / simulations,
        "67_percent": succes_67_percent / simulations,
        "optimale": succes_optimal / simulations
    }




#%% Test Simulation Strategies

max_n = 50
simulations = 10000

ns = range(1, max_n + 1)
prob_aleatoire = []
prob_50_percent = []
prob_80_percent = []
prob_25_percent = []
prob_67_percent = []
prob_optimale = []

for N in ns:
    resultats = comparer_strategies(N, simulations)
    prob_aleatoire.append(resultats["aleatoire"])
    prob_50_percent.append(resultats["50_percent"])
    prob_80_percent.append(resultats["80_percent"])
    prob_25_percent.append(resultats["20_percent"])
    prob_67_percent.append(resultats["67_percent"])
    prob_optimale.append(resultats["optimale"])
    

#%% plot

plt.figure(figsize=(14, 6))
plt.plot(ns, prob_aleatoire, marker='.', color='orange', label='Stratégie Aléatoire')
plt.plot(ns, prob_25_percent, marker='.', color='lightgrey', label='Stratégie 20%')
plt.plot(ns, prob_50_percent, marker='.', color='lightblue', label='Stratégie 50%')
plt.plot(ns, prob_67_percent, marker='.', color='blue', label='Stratégie 67%')
plt.plot(ns, prob_80_percent, marker='.', color='darkblue', label='Stratégie 80%')
plt.plot(ns, prob_optimale, marker='.', color='green', label='Stratégie Optimale (39%)')
plt.xlabel('Nombre de Candidats (N)')
plt.ylabel('Probabilité de Choisir le Meilleur Candidat')
plt.title('Probabilité de Sélectionner le Meilleur Candidat vs N')
plt.axhline(y=np.exp(-1), color='r', linestyle='--', label='1 / e ')
plt.legend()
plt.tight_layout()
plt.show()




