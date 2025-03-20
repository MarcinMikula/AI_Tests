import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from AI.jira.jira_utils import report_success_to_jira, report_defect_to_jira
from Algorytmy.config import ISSUE_KEY, DB_PATH


# Konfiguracja bazy danych (SQLite)
print(f"Using database path: {DB_PATH}")
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
Session = sessionmaker(bind=engine)
session = Session()

# Funkcja pobierania opisu typu testu z bazy
def get_test_type_info(test_type):
    test_type_record = session.query(TestType).filter_by(test_type=test_type).first()
    if test_type_record:
        return {
            "goal": test_type_record.goal,
            "example": test_type_record.example,
            "test_data": test_type_record.test_data,
            "tools_used": test_type_record.tools_used,
            "data_manipulation": test_type_record.data_manipulation
        }
    return {"goal": "N/A", "example": "N/A", "test_data": "N/A", "tools_used": "N/A", "data_manipulation": "N/A"}

# Test Data Quality
tf.random.set_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Obliczenie rozkładu klas w danych treningowych i testowych
train_class_counts = np.bincount(y_train)
test_class_counts = np.bincount(y_test)

# Sprawdzenie balansu (odchylenie standardowe liczby próbek na klasę)
train_std = np.std(train_class_counts)
test_std = np.std(test_class_counts)

THRESHOLD_STD = 100  # Maksymalne dopuszczalne odchylenie standardowe (MNIST jest dość zbalansowany)
balanced = train_std <= THRESHOLD_STD and test_std <= THRESHOLD_STD
TEST_SIZE = len(x_test)

model_details = "No model training – Data Quality Test"
class_dist = ", ".join([f"Class {i}: {np.sum(y_test == i)}" for i in range(10)])

# Zapis do bazy
test = session.query(Test).filter_by(issue_key=ISSUE_KEY).first()
if not test:
    test = Test(
        issue_key=ISSUE_KEY,
        test_name="WYM-006 Data Quality 01",
        test_type="Data Quality",
        requirement="WYM-006",
        test_number="01"
    )
    session.add(test)
    session.commit()

result = TestResult(
    test_id=test.id,
    accuracy=0.0,  # Brak modelu – ustawiamy 0
    execution_time=0.0,
    model_details=model_details,
    class_distribution=class_dist,
    status="Passed" if balanced else "Failed"
)
session.add(result)
session.commit()

# Raport do JIRA
test_type_info = get_test_type_info("Danych")
if balanced:
    report_success_to_jira(ISSUE_KEY, 0.0, "Data Quality", test_type_info)
else:
    # Dla Data Quality nie mamy y_pred, więc przekazujemy dummy dane dla report_defect_to_jira
    dummy_y_true = np.zeros(TEST_SIZE)
    dummy_y_pred = np.zeros(TEST_SIZE)
    defect_key, precision, recall, f1, training_details, environment, top_misclassification = report_defect_to_jira(
        result.id, 0.0, THRESHOLD_STD, ISSUE_KEY, dummy_y_true, dummy_y_pred, None, 0, TEST_SIZE, 0.0, model_details, class_dist
    )
    if defect_key:
        defect = Defect(
            test_result_id=result.id,
            issue_key=defect_key,
            description=f"Data Quality Test: Unbalanced data - Train STD={train_std:.2f}, Test STD={test_std:.2f}",
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            training_details="N/A",
            environment="N/A",
            top_misclassification="N/A"
        )
        session.add(defect)
        session.commit()

session.close()