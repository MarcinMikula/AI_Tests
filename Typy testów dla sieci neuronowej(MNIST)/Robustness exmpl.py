'''
Testowanie odporności (Robustness Testing)
Cel: Ocena, jak model radzi sobie z zakłóconymi lub nietypowymi danymi wejściowymi (np. szum, ataki adversarialne).
Podejście: Dodajemy szum do obrazu i sprawdzamy, czy model nadal poprawnie klasyfikuje.
Przykład:
Dodajemy losowy szum.
'''

import numpy as np
noisy_image = x_test[0] + np.random.normal(0, 0.1, x_test[0].shape)  # Szum gaussowski
noisy_image = np.clip(noisy_image, 0, 1)  # Ograniczenie do [0, 1]
noisy_image = noisy_image.reshape(1, 28 * 28)
prediction = model.predict(noisy_image)
predicted_digit = np.argmax(prediction)
print(f"Robustness test: Predicted {predicted_digit} for noisy image (expected {y_test[0]})")

'''
Wynik: Model powinien nadal przewidzieć "7", choć przy większym szumie może się pomylić – testujemy granicę odporności.
'''