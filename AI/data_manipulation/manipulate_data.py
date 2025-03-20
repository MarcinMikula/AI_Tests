import numpy as np
from scipy.ndimage import rotate


def add_noise(data, noise_factor=0.1):
    """
    Dodaje szum gaussowski do danych.
    Args:
        data: Tablica NumPy z danymi (np. obrazy Fashion MNIST w formacie (n_samples, 784))
        noise_factor: Siła szumu (std odchylenia standardowego)
    Returns:
        Tablica NumPy z zaszumionymi danymi
    """
    noisy_data = data + np.random.normal(0, noise_factor, data.shape)
    noisy_data = np.clip(noisy_data, 0, 1)
    return noisy_data


def unbalance_classes(x, y, target_class, target_ratio=0.9):
    """
    Tworzy niezbalansowany zbiór danych, gdzie target_class stanowi target_ratio danych.
    Args:
        x: Tablica NumPy z danymi (np. obrazy Fashion MNIST w formacie (n_samples, 784))
        y: Tablica NumPy z etykietami (np. (n_samples,))
        target_class: Klasa, która ma dominować (np. 0 dla 'T-shirt/top')
        target_ratio: Proporcja danych dla target_class (np. 0.9 = 90%)
    Returns:
        Tuple (x_unbalanced, y_unbalanced) – niezbalansowane dane i etykiety
    """
    # Oddziel dane dla target_class
    mask_target = (y == target_class)
    x_target = x[mask_target]
    y_target = y[mask_target]

    # Oddziel dane dla innych klas
    mask_others = (y != target_class)
    x_others = x[mask_others]
    y_others = y[mask_others]

    # Oblicz, ile próbek potrzebujemy dla target_class
    total_samples = len(x)
    target_count = int(total_samples * target_ratio)
    other_count = total_samples - target_count

    # Losowe wybieranie próbek
    if len(x_target) < target_count:
        # Jeśli za mało próbek, duplikujemy
        indices_target = np.random.choice(len(x_target), target_count, replace=True)
    else:
        indices_target = np.random.choice(len(x_target), target_count, replace=False)

    if len(x_others) < other_count:
        indices_others = np.random.choice(len(x_others), other_count, replace=True)
    else:
        indices_others = np.random.choice(len(x_others), other_count, replace=False)

    # Połączenie danych
    x_unbalanced = np.vstack((x_target[indices_target], x_others[indices_others]))
    y_unbalanced = np.hstack((y_target[indices_target], y_others[indices_others]))

    return x_unbalanced, y_unbalanced


def adjust_brightness(data, factor=0.8):
    """
    Zmienia jasność danych (obrazów).
    Args:
        data: Tablica NumPy z danymi (np. obrazy Fashion MNIST w formacie (n_samples, 784))
        factor: Czynnik jasności (np. 0.8 = przyciemnienie o 20%)
    Returns:
        Tablica NumPy z danymi o zmienionej jasności
    """
    bright_data = data * factor
    bright_data = np.clip(bright_data, 0, 1)
    return bright_data


def rotate_images(data, angle=10):
    """
    Obraca obrazy o zadany kąt.
    Args:
        data: Tablica NumPy z danymi (np. obrazy Fashion MNIST w formacie (n_samples, 784))
        angle: Kąt obrotu w stopniach
    Returns:
        Tablica NumPy z obróconymi obrazami
    """
    # Przekształcenie danych do formatu 2D (28x28)
    data_2d = data.reshape(-1, 28, 28)
    # Obrót każdego obrazu
    rotated_data = np.array([rotate(image, angle, reshape=False) for image in data_2d])
    # Powrót do formatu 1D (784)
    rotated_data = rotated_data.reshape(-1, 784)
    rotated_data = np.clip(rotated_data, 0, 1)
    return rotated_data