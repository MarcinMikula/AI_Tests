'''
Testowanie ciągłej integracji (CI Testing)
Cel: Zapewnienie, że model działa poprawnie w pipeline CI po zmianach.

Podejście: Automatyzacja testu funkcjonalnego w CI (np. Jenkins).

Przykład:
Skrypt do CI sprawdzający minimalną dokładność.
'''

import sys
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
if test_accuracy < 0.95:
    print(f"CI test failed: Accuracy {test_accuracy:.4f} < 0.95")
    sys.exit(1)
print("CI test passed")

'''
Wynik: Kod zwraca błąd (exit 1), jeśli accuracy < 95%, co zatrzyma pipeline CI.
'''