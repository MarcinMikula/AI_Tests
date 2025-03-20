'''
Testowanie danych (Data Quality Testing)
Cel: Weryfikacja jakości danych wejściowych (np. brakujące wartości, niezbalansowanie).

Podejście: Sprawdzamy, czy dane testowe są kompletne i reprezentatywne.

Przykład:
Liczymy rozkład klas w y_train.
'''

unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"Data quality test: Class distribution = {class_distribution}")
for count in counts:
    assert count > 5000, "Class imbalance detected!"

'''
Wynik: MNIST jest dobrze zbalansowany (~6000 przykładów na klasę), ale w rzeczywistych danych moglibyśmy wykryć problem.
'''