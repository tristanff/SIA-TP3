import pandas as pd
import numpy as np
from perceptron import LinealPerceptron, HiperbolicPerceptron, LogisticPerceptron

# Fonction pour obtenir les données
def get_data():
    data = pd.read_csv('TP3-ej2-conjunto.csv')  # Changez cela avec votre fichier CSV
    input_data = np.array(data[['x1', 'x2', 'x3']])  # Variables d'entrée
    expected_data = np.array(data['y'])  # Sorties attendues
    return input_data, expected_data

# Obtenir les données
input_data, expected_data = get_data()

# Paramètres
train_ratio = 0.8  # Ratio de formation fixe
split_index = int(len(input_data) * train_ratio)
training_data = input_data[:split_index].tolist()  # Données d'entraînement
testing_data = input_data[split_index:].tolist()  # Données de test

# Hyperparamètres
input_size = 3
bias = 1
error_threshold = 0.0001
max_epochs = 5000
learning_rate = 0.01  # Taux d'apprentissage fixe
betas = [0.1, 0.5, 1, 2, 5]  # Liste des valeurs de beta
num_iterations = 10  # Nombre d'itérations pour la moyenne

# Fonction utilitaire pour étendre les tableaux à la même longueur
def extend_to_max_length(values, max_length):
    return np.pad(values, (0, max_length - len(values)), 'edge')

# Fonction pour exécuter les expériences avec différentes valeurs de beta pour différents perceptrons
def run_experiments_with_betas(filename):
    results = []

    for beta in betas:
        # Pour le Perceptron Linéaire
        all_mse = []
        max_length = 0
        for _ in range(num_iterations):
            linear_perceptron = LinealPerceptron(learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = linear_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs,scale=True
            )
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Linear',
                'Beta': beta,
                'Epoch': epoch,
                'MSE': mse
            })

        # Pour le Perceptron Hyperbolique
        all_mse = []
        max_length = 0
        for _ in range(num_iterations):
            hiperbolic_perceptron = HiperbolicPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = hiperbolic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs,scale=False
            )
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Hyperbolic',
                'Beta': beta,
                'Epoch': epoch,
                'MSE': mse
            })

        # Pour le Perceptron Logistique
        all_mse = []
        max_length = 0
        for _ in range(num_iterations):
            logistic_perceptron = LogisticPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = logistic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs,scale=False
            )
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Logistic',
                'Beta': beta,
                'Epoch': epoch,
                'MSE': mse
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)  # Enregistrer dans un CSV

# Exécuter les expériences et sauvegarder les résultats
csv_filename = 'MSE_byBeta-Epochs.csv'
#run_experiments_with_betas(csv_filename)

import pandas as pd
import matplotlib.pyplot as plt

# Lire les résultats du CSV
results_df = pd.read_csv('MSE_byBeta-Epochs.csv')

perceptron_types = results_df['Perceptron Type'].unique()
betas = results_df['Beta'].unique()

# Create a plot for each perceptron type
for perceptron_type in perceptron_types:
    plt.figure(figsize=(10, 6))

    # Filter data by perceptron type
    perceptron_results = results_df[results_df['Perceptron Type'] == perceptron_type]

    # Plot MSE for each beta value
    for beta in betas:
        # Filter data for the specific beta
        beta_results = perceptron_results[perceptron_results['Beta'] == beta]

        # Plotting
        plt.plot(beta_results['Epoch'], beta_results['MSE'], label=f'Beta = {beta}')

    # Adding labels and title
    plt.xlabel('Époque')
    plt.ylabel('MSE')
    plt.title(f'MSE by Number of Epochs for {perceptron_type}')
    plt.legend()

    # Show the plot
    plt.show()

