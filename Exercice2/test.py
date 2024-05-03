import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from perceptron import LinearPerceptron, HyperbolicPerceptron, LogisticPerceptron
from sklearn.model_selection import train_test_split

# Charger les données
data = pd.read_csv("TP3-ej2-conjunto.csv")  # Assurez-vous que le fichier CSV est accessible
x = data[["x1", "x2", "x3"]].values  # Les entrées
y = data["y"].values  # Les sorties attendues

# Paramètres de base
learning_rate = 0.1  # Taux d'apprentissage
bias = 0.0  # Biais initial
epsilon = 0.5  # Tolérance d'erreur
max_epochs = 1000  # Nombre maximal d'époques
# Définition des types de perceptrons disponibles
PERCEPTRON_TYPES = {
    "Linear": LinearPerceptron,
    "Hyperbolic": HyperbolicPerceptron,
    "Logistic": LogisticPerceptron,
}

REPEATS = 10  # Nombre de répétitions


# Fonction pour entraîner le perceptron et retourner les résultats
def run_main(perceptron_class, learning_rate, bias, input_size, epsilon, train_percentage):
    # Diviser les données en ensembles d'entraînement et de test

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_percentage), random_state=42)

    # Créer une instance du perceptron avec tous les arguments requis
    perceptron = perceptron_class(learning_rate, None, bias, input_size, epsilon)

    # Entraîner le perceptron
    epochs, train_errors, test_errors = perceptron.train(x_train, x_test, y_train, y_test, max_epochs)

    # Retourner les résultats
    return epochs, test_errors[-1], test_errors


# Paramètres pour le perceptron
learning_rate = 0.1  # Taux d'apprentissage
bias = 0.0  # Biais initial
input_size = len(x[0])  # Taille des entrées
epsilon = 0.001  # Tolérance d'erreur
max_epochs = 300 # Nombre maximal d'époques

result_list = []
REPEATS = 10  # Nombre de répétitions

# Types de perceptrons disponibles
PERCEPTRON_TYPES = {
    "Linear": LinearPerceptron,
    "Hyperbolic": HyperbolicPerceptron,
    "Logistic": LogisticPerceptron,
}

# Boucle sur les types de perceptrons et les pourcentages d'entraînement
for perceptron_name, perceptron_class in PERCEPTRON_TYPES.items():
    for tp in [0.3, 0.5, 0.7, 0.9]:  # Pourcentages d'entraînement
        for run in range(1, REPEATS + 1):
            epochs, mse, test_errors = run_main(perceptron_class, learning_rate, bias, input_size, epsilon, tp)
            result_list.append({
                "perceptron_type": perceptron_name,
                "training_percentage": tp,
                "mse": mse,
                "run": run,
            })

# Convertir les résultats en DataFrame pour analyse
df = pd.DataFrame(result_list)

# Afficher les graphiques
import matplotlib.pyplot as plt

colors = plt.cm.viridis(np.linspace(0, 1, 4))  # Couleurs pour les graphiques

# Boucle pour tracer les résultats par type de perceptron
for perceptron_name in PERCEPTRON_TYPES.keys():
    subset = df[df["perceptron_type"] == perceptron_name]

    plt.figure(figsize=(10, 6))  # Taille de la figure

    i = 0
    for tp in subset["training_percentage"].unique():  # Boucle sur les pourcentages
        tp_subset = subset[subset["training_percentage"] == tp]
        mean_tp_subset = tp_subset["mse"].mean()  # Moyenne du MSE
        std_error = tp_subset["mse"].std()  # Écart type

        # Tracer la barre avec l'erreur moyenne
        plt.bar(tp, mean_tp_subset, capsize=4, label=f"Training Percentage: {tp}", width=0.1, color=colors[i])

        # Ajouter du texte avec la valeur moyenne
        plt.text(tp, mean_tp_subset, f"{mean_tp_subset:.4f}", ha='center', va='bottom')
        i += 1  # Incrémenter l'index pour la couleur

    # Étiquettes et titre
    plt.xlabel('Training Percentage')
    plt.ylabel('Mean Squared Error')
    plt.title(f'MSE vs. Training Percentage for "{perceptron_name}"')
    plt.legend(title='Training Percentage')

    # Afficher la grille
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()  # Afficher le graphique
