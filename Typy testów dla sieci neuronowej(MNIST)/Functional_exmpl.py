'''
 Testowanie funkcjonalne (Functional Testing)
Cel: Sprawdzenie, czy sieć neuronowa spełnia swoje wymagania funkcjonalne, czyli poprawnie klasyfikuje cyfry.
Podejście: Porównanie predykcji modelu z oczekiwanymi wynikami na znanym zestawie danych.
Przykład:
Sprawdzamy, czy model poprawnie rozpoznaje cyfrę "7" z x_test[0].
'''


sample_image = x_test[0].reshape(1, 28 * 28)
prediction = model.predict(sample_image)
predicted_digit = np.argmax(prediction)
expected_digit = y_test[0]  # Oczekiwana cyfra (7)
assert predicted_digit == expected_digit, f"Expected {expected_digit}, but got {predicted_digit}"
print(f"Functional test passed: Predicted {predicted_digit}, Expected {expected_digit}")

'''
Wynik: Jeśli predicted_digit = 7 i y_test[0] = 7, test przechodzi.
'''