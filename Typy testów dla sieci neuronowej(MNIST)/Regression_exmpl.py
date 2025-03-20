'''
Testowanie regresji (Regression Testing)

Cel: Sprawdzenie, czy nowa wersja modelu nie pogorszyła wyników w porównaniu do poprzedniej.

Podejście: Porównanie accuracy przed i po zmianie (np. zwiększeniu epok).

Przykład:
Trenujemy model na 10 epok i porównujemy.
'''

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)  # Dodatkowe 5 epok
new_loss, new_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Regression test: New accuracy = {new_accuracy:.4f}, Previous = {test_accuracy:.4f}")
assert new_accuracy >= test_accuracy, "Regression detected!"

'''
Wynik: Jeśli nowa dokładność jest niższa, coś poszło nie tak (np. przetrenowanie).
'''