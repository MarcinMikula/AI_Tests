'''
Cel: Ocena metryk jakości modelu (np. accuracy, precision, recall) na danych testowych.
Podejście: Używamy wbudowanej metody evaluate i analizujemy wyniki.
Przykład:
Sprawdzamy dokładność modelu na całym zestawie testowym.
'''

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
assert test_accuracy > 0.95, f"Accuracy {test_accuracy} is below 95%"
print(f"Performance test: Test accuracy = {test_accuracy:.4f}")

'''
Wynik: Oczekujemy accuracy > 95% (np. 0.9750). Jeśli jest niższe, model wymaga poprawy.
'''
