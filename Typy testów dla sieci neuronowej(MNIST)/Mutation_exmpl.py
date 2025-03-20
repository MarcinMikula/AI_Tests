'''
Cel: Ocena jakości testów poprzez modyfikację modelu lub danych.

Podejście: Wprowadzamy "mutację" (np. losowe zmiany wag) i sprawdzamy, czy testy to wykrywają.

Przykład:
Mutujemy wagi i testujemy wydajność.
'''

original_weights = model.layers[0].get_weights()
mutated_weights = [w + np.random.normal(0, 0.1, w.shape) for w in original_weights]
model.layers[0].set_weights(mutated_weights)
mutated_loss, mutated_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Mutation test: Mutated accuracy = {mutated_accuracy:.4f} (original {test_accuracy:.4f})")

'''
Wynik: Spadek accuracy (np. z 0.97 na 0.90) wskazuje, że testy są wrażliwe na zmiany.
'''