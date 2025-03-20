import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Ustalenie absolutnej ścieżki do bazy w głównym katalogu
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Wychodzimy o jeden katalog w górę
DB_PATH = os.path.join(BASE_DIR, 'test_results.db')
print(f"Using database path: {DB_PATH}")

# Usunięcie starej bazy (opcjonalne – możesz to zakomentować, jeśli chcesz zachować dane)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print("Old database removed – starting fresh.")

# Konfiguracja bazy danych (SQLite)
engine = create_engine(f'sqlite:///{DB_PATH}', echo=True)
Base = declarative_base()

# Definicja tabel
class Test(Base):
    __tablename__ = 'tests'
    id = Column(Integer, primary_key=True)
    issue_key = Column(String, unique=True, nullable=False)
    test_name = Column(String, nullable=False)
    test_type = Column(String, nullable=False)
    requirement = Column(String, nullable=False)
    test_number = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class TestResult(Base):
    __tablename__ = 'test_results'
    id = Column(Integer, primary_key=True)
    test_id = Column(Integer, ForeignKey('tests.id'), nullable=False)
    accuracy = Column(Float, nullable=False)
    execution_time = Column(Float, nullable=False)
    model_details = Column(Text, nullable=False)
    class_distribution = Column(Text, nullable=False)
    executed_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, nullable=False)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    test_result_id = Column(Integer, ForeignKey('test_results.id'), nullable=False)
    y_true = Column(Text, nullable=False)
    y_pred = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Defect(Base):
    __tablename__ = 'defects'
    id = Column(Integer, primary_key=True)
    test_result_id = Column(Integer, ForeignKey('test_results.id'), nullable=False)
    issue_key = Column(String, nullable=False)
    description = Column(String, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    training_details = Column(Text, nullable=False)
    environment = Column(Text, nullable=False)
    top_misclassification = Column(Text, nullable=False)
    reported_at = Column(DateTime, default=datetime.utcnow)

class ConfusionMatrix(Base):
    __tablename__ = 'confusion_matrices'
    id = Column(Integer, primary_key=True)
    test_result_id = Column(Integer, ForeignKey('test_results.id'), nullable=False)
    matrix_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

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

# Tworzenie tabel
Base.metadata.create_all(engine)
print("Database tables created successfully.")