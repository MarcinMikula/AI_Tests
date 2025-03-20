'''
Testowanie etyczności (Ethical Testing)

Cel: Ocena, czy model nie wykazuje uprzedzeń (bias) lub nieetycznych zachowań.
Podejście: Analiza predykcji dla różnych podzbiorów danych (np. czy model gorzej rozpoznaje cyfry pisane w nietypowy sposób).

Przykład:
Sprawdzamy dokładność dla cyfry "9" (która może być mylona z "4").
'''

mask = (y_test == 9)  # Wybór wszystkich "9"
x_nines = x_test[mask]
y_nines = y_test[mask]
loss, accuracy = model.evaluate(x_nines, y_nines, verbose=0)
print(f"Ethical test: Accuracy for digit 9 = {accuracy:.4f}")

'''
Wynik: Jeśli accuracy dla "9" jest znacznie niższe niż średnia (np. 0.85 vs 0.97), może wskazywać na bias w danych treningowych.
'''