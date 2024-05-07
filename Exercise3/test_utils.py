import numpy as np
from noise_utils import salt_pepper_noise, random_noise, gaussian_noise

def generate_test_dataset(num_values, data_source, noise_type=None, salt_probability=0.1, pepper_probability=0.1, mean=0, std=0.1, factor=10):
    dataX = []
    dataY = []
    
    for i in range(num_values):
        index = np.random.randint(10)
        digit_img = data_source[index]
        
        if noise_type == 'salt_pepper':
            digit_img = salt_pepper_noise(digit_img, salt_probability, pepper_probability)
        elif noise_type == 'random':
            digit_img = random_noise(digit_img, factor)
        elif noise_type == 'gauss':
            digit_img = gaussian_noise(digit_img, mean, std)           
        
        dataX.append(digit_img)
        dataY.append(index)

    return np.array(dataX).squeeze(), np.array(dataY)

def train_test(dataX, dataY, num_values, percentage):
    split = round(num_values*percentage)
    
    dataX_train = dataX[:split]
    dataY_train = dataY[:split]
    dataX_test = dataX[split:]
    dataY_test = dataY[split:]
    
    return dataX_train, dataY_train, dataX_test, dataY_test

def accuracy(dataX_test, dataY_test):
    test_list = dataX_test == dataY_test
    accuracy = (len([ele for ele in test_list if ele == True]) / len(test_list))
    return accuracy