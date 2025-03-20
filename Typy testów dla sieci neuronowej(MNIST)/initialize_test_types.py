import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Połączenie z istniejącą bazą
engine = create_engine('sqlite:///test_results.db', echo=True)
Base = declarative_base()

# Definicja tabeli TEST_TYPES
class TestType(Base):
    __tablename__ = 'test_types'
    id = Column(Integer, primary_key=True)
    test_type = Column(String, nullable=False, unique=True)
    goal = Column(Text, nullable=False)
    example = Column(Text, nullable=False)
    test_data = Column(Text, nullable=False)
    tools_used = Column(Text, nullable=False)
    data_manipulation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Tworzenie tabeli (jeśli nie istnieje)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Dane do zapisania
test_types_data = [
    {
        "test_type": "Funkcjonalne",
        "goal": "Sprawdzenie, czy model poprawnie wykonuje swoje podstawowe zadanie – w tym przypadku klasyfikację obrazów cyfr. Testy funkcjonalne weryfikują, czy model przewiduje poprawną klasę dla danego obrazu wejściowego, zgodnie z oczekiwaniami opisanymi w wymaganiach.",
        "example": "Predykcja '7' dla obrazu",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Obrazy MNIST bez modyfikacji",
        "data_manipulation": "Brak – dane oryginalne"
    },
    {
        "test_type": "Odporności",
        "goal": "Ocena zdolności modelu do poprawnego działania w obecności zakłóceń danych wejściowych, takich jak szum, zmiany jasności czy obrót. Testy odporności sprawdzają, czy model jest stabilny i odporny na nieprzewidziane warunki, co jest kluczowe w rzeczywistych zastosowaniach.",
        "example": "Klasyfikacja zaszumionego obrazu",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Obrazy MNIST z dodanym szumem",
        "data_manipulation": "Dodanie szumu gaussowskiego (std=0.1)"
    },
    {
        "test_type": "Wydajności",
        "goal": "Pomiar kluczowych metryk wydajności, takich jak dokładność (accuracy), precyzja (precision), czułość (recall) czy F1-score. Testy wydajności weryfikują, czy model osiąga akceptowalny poziom skuteczności na danych testowych, zgodnie z zadanymi progami (np. accuracy > 95%).",
        "example": "Accuracy > 95% na danych testowych",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Obrazy MNIST bez modyfikacji",
        "data_manipulation": "Brak – dane oryginalne"
    },
    {
        "test_type": "Wyjaśnialności",
        "goal": "Zrozumienie, dlaczego model podejmuje dane decyzje i które cechy danych wejściowych mają największy wpływ na predykcje. Testy wyjaśnialności pomagają budować zaufanie do modelu i diagnozować potencjalne problemy, np. czy model skupia się na odpowiednich pikselach obrazu.",
        "example": "Analiza ważnych pikseli",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Wagi pierwszej warstwy modelu",
        "data_manipulation": "Analiza wag lub metoda SHAP"
    },
    {
        "test_type": "Etyczności",
        "goal": "Weryfikacja, czy model nie wykazuje uprzedzeń (bias) wobec określonych klas lub danych, co mogłoby prowadzić do nierównego traktowania. Testy etyczności sprawdzają, czy model osiąga porównywalną skuteczność dla wszystkich klas (np. czy cyfra '9' nie jest gorzej rozpoznawana).",
        "example": "Accuracy dla cyfry '9'",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Podzbiór danych dla klasy '9'",
        "data_manipulation": "Analiza metryk per klasa"
    },
    {
        "test_type": "Danych",
        "goal": "Ocena jakości i reprezentatywności danych wejściowych używanych do treningu i testowania modelu. Testy danych sprawdzają, czy dane są zbalansowane, kompletne i wolne od błędów, co ma kluczowy wpływ na wydajność modelu.",
        "example": "Rozkład klas w danych",
        "test_data": "Zbiór treningowy MNIST",
        "tools_used": "Etykiety klas (y_train)",
        "data_manipulation": "Obliczanie histogramu klas"
    },
    {
        "test_type": "Regresji",
        "goal": "Zapewnienie, że zmiany w modelu (np. retrening, zmiana hiperparametrów) nie powodują pogorszenia wydajności w porównaniu do wcześniejszej wersji. Testy regresji porównują metryki (np. accuracy) przed i po zmianach.",
        "example": "Porównanie accuracy po retreningu",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Model przed i po retreningu",
        "data_manipulation": "Retraining z nowymi hiperparametrami"
    },
    {
        "test_type": "CI",
        "goal": "Integracja testów w procesie ciągłej integracji (CI), by automatycznie weryfikować model przy każdej zmianie kodu lub danych. Testy CI zapewniają, że model nadal spełnia minimalne wymagania (np. accuracy > 95%).",
        "example": "Test minimalnej dokładności",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Pipeline CI (np. Jenkins)",
        "data_manipulation": "Automatyczne uruchamianie testów"
    },
    {
        "test_type": "Mutacyjne",
        "goal": "Ocena jakości samych testów poprzez wprowadzanie celowych zmian (mutacji) w modelu (np. losowe zmiany wag) i sprawdzanie, czy testy wykrywają te zmiany poprzez spadek metryk wydajności. Testy mutacyjne weryfikują, czy zestaw testów jest wystarczająco czuły.",
        "example": "Spadek accuracy po mutacji wag",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Wagi modelu",
        "data_manipulation": "Losowa zmiana wag (mutacja)"
    },
    {
        "test_type": "Obciążeniowe",
        "goal": "Sprawdzenie, jak model radzi sobie z dużym obciążeniem danych, np. przy przetwarzaniu dużej liczby obrazów w krótkim czasie. Testy obciążeniowe mierzą czas predykcji i stabilność modelu pod presją.",
        "example": "Czas predykcji dla 10 000 obrazów",
        "test_data": "Zbiór testowy MNIST",
        "tools_used": "Duża liczba obrazów (np. 10 000)",
        "data_manipulation": "Przetwarzanie wsadowe"
    },
]

# Zapis danych do bazy
for test_type_data in test_types_data:
    # Sprawdzenie, czy typ testu już istnieje
    existing_test_type = session.query(TestType).filter_by(test_type=test_type_data["test_type"]).first()
    if not existing_test_type:
        test_type = TestType(
            test_type=test_type_data["test_type"],
            goal=test_type_data["goal"],
            example=test_type_data["example"],
            test_data=test_type_data["test_data"],
            tools_used=test_type_data["tools_used"],
            data_manipulation=test_type_data["data_manipulation"]
        )
        session.add(test_type)

session.commit()
session.close()
print("Test types initialized in the database.")