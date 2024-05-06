import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import LinealPerceptron, HiperbolicPerceptron, LogisticPerceptron

# Fonction pour obtenir les données et les diviser en ensembles d'entraînement et de test
def get_data():
    data = pd.read_csv('TP3-ej2-conjunto.csv')  # Remplacez avec votre fichier CSV
    input_data = np.array(data[['x1', 'x2', 'x3']])  # Variables d'entrée
    expected_data = np.array(data['y'])  # Sorties attendues
    return input_data, expected_data

input_data, expected_data = get_data()
train_ratio = 0.8
split_index = int(len(input_data) * train_ratio)
training_data = input_data[:split_index].tolist()
testing_data = input_data[split_index:].tolist()

# Paramètres
input_size = 3
learning_rate = 0.01  # Taux d'apprentissage fixe
bias = 1  # Biais fixé
max_epochs = 10000  # Maximum d'époques pour l'entraînement
error_thresholds = [0.05,0.04,0.03,0.02,0.01]  # Seuils d'erreur à tester
num_iterations = 100  # Nombre d'itérations pour la moyenne

# Fonction pour exécuter des expériences avec des seuils d'erreur variables
def run_experiments_with_error_thresholds(filename):
    results = []
    for error_threshold in error_thresholds:
        # Pour le Perceptron Linéaire
        total_epochs = []
        for _ in range(num_iterations):
            linear_perceptron = LinealPerceptron(learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = linear_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            total_epochs.append(epochs)

        avg_epochs = np.mean(np.array(total_epochs))

        results.append({
            'Perceptron Type': 'Linear',
            'Error Threshold': error_threshold,
            'Average Epochs': avg_epochs
        })

        # Pour le Perceptron Hyperbolique
        total_epochs = []
        for _ in range(num_iterations):
            hiperbolic_perceptron = HiperbolicPerceptron(1, learning_rate, [], bias, input_size, error_threshold)  # Beta fixé à 1
            epochs, train_errors, test_errors = hiperbolic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            total_epochs.append(epochs)

        avg_epochs = np.mean(np.array(total_epochs))

        results.append({
            'Perceptron Type': 'Hyperbolic',
            'Error Threshold': error_threshold,
            'Average Epochs': avg_epochs
        })
        # Pour le Perceptron Logistique
        total_epochs = []
        for _ in range(num_iterations):
            logistic_perceptron = LogisticPerceptron(1, learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = logistic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            total_epochs.append(epochs)

        avg_epochs = np.mean(np.array(total_epochs))

        results.append({
            'Perceptron Type': 'Logistic',
            'Error Threshold': error_threshold,
            'Average Epochs': avg_epochs
        })

    # Enregistrer les résultats dans un fichier CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)

# Exécuter les expériences et enregistrer les résultats
csv_filename = 'Epochs_byErrorThreshold.csv'
run_experiments_with_error_thresholds(csv_filename)

# Lire les résultats et créer un graphique à barres pour afficher le nombre moyen d'époques pour chaque seuil d'erreur
results_df = pd.read_csv(csv_filename)

# Perceptron types
perceptron_types = results_df['Perceptron Type'].unique()

# Créer un graphique à barres pour chaque perceptron
for perceptron_type in perceptron_types:
    plt.figure(figsize=(10, 6))

    # Filtrer les données par type de perceptron
    perceptron_results = results_df[results_df['Perceptron Type'] == perceptron_type]

    # Tracer un graphique à barres du nombre moyen d'époques pour chaque seuil d'erreur
    plt.bar(
        perceptron_results['Error Threshold'].astype(str),
        perceptron_results['Average Epochs'],
        color='b',
        alpha=0.7
    )

    # Ajouter des étiquettes et un titre
    plt.xlabel('Error Threshold')
    plt.ylabel('Average Epochs')
    plt.title(f'Average Epochs by Error Threshold for {perceptron_type}')

    # Afficher le graphique
    plt.show()
