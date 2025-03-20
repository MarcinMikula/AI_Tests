'''
Testowanie wyjaśnialności (Explainability Testing)
Cel: Sprawdzenie, czy można zrozumieć, dlaczego model podejmuje dane decyzje.
Podejście: Użycie technik jak LIME lub analiza aktywacji warstw, by zobaczyć, które piksele wpływają na predykcję.
Przykład:
Prosty test: Sprawdzamy wagi pierwszej warstwy, by zobaczyć, które piksele są ważne.
'''

import numpy as np

weights = model.layers[0].get_weights()[0]  # Wagi pierwszej warstwy
important_pixels = np.mean(np.abs(weights), axis=1)  # Średnia ważność pikseli
print(f"Explainability test: Top 5 influential pixels: {np.argsort(important_pixels)[-5:]}")

'''
Wynik: Zwraca indeksy pikseli (0-783), które mają największy wpływ na predykcje – można je zwizualizować jako heatmapę.
'''