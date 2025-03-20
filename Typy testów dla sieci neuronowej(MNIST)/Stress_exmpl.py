'''
Testowanie obciążeniowe (Stress Testing)
Cel: Sprawdzenie, jak model radzi sobie z ekstremalnymi danymi lub obciążeniem.

Podejście: Testujemy na bardzo dużych batchach lub zdegenerowanych danych.

Przykład:
Predykcja na 10 000 obrazów naraz.
'''

import time
start_time = time.time()
predictions = model.predict(x_test)
elapsed_time = time.time() - start_time
print(f"Stress test: Time for 10,000 predictions = {elapsed_time:.2f} seconds")

'''
Wynik: Mierzymy czas – jeśli za długi (np. >10s na słabym sprzęcie), może być problem z wydajnością.
'''