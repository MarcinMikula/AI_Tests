import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from sklearn.metrics import confusion_matrix
import time
from AI.jira.jira_utils import report_defect_to_jira, report_success_to_jira
import json
from Algorytmy.config import ISSUE_KEY, DB_PATH
from AI.data_manipulation.manipulate_data import add_noise  # Import funkcji manipulacji

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
Session = sessionmaker(bind=engine)
session = Session()

# Funkcja pobierania opisu typu testu z bazy
def get_test_type_info(test_type):
    all_types = session.query(TestType).all()
    print("Available test types in TEST_TYPES table:")
    for t in all_types:
        print(f"- {t.test_type}")

    test_type_record = session.query(TestType).filter(TestType.test_type.ilike(test_type)).first()
    if test_type_record:
        print(f"Found test type: {test_type_record.test_type}")
        return {
            "goal": test_type_record.goal,
            "example": test_type_record.example,
            "test_data": test_type_record.test_data,
            "tools_used": test_type_record.tools_used,
            "data_manipulation": test_type_record.data_manipulation
        }
    else:
        print(f"Test type {test_type} not found in TEST_TYPES table.")
        return {"goal": "N/A", "example": "N/A", "test_data": "N/A", "tools_used": "N/A", "data_manipulation": "N/A"}

# Test Robustness
tf.random.set_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 5

start_time = time.time()
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, verbose=1)
execution_time = time.time() - start_time

# Test na oryginalnych danych
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
y_pred_clean = np.argmax(model.predict(x_test), axis=1)

# Test Robustness: Dodajemy szum gaussowski za pomocą funkcji z manipulate_data
x_test_noisy = add_noise(x_test, noise_factor=0.1)
test_loss_noisy, test_accuracy_noisy = model.evaluate(x_test_noisy, y_test, verbose=0)
y_pred_noisy = np.argmax(model.predict(x_test_noisy), axis=1)

THRESHOLD = 0.99  # Zmieniony próg
TEST_SIZE = len(x_test)

model_details = f"Layers={len(model.layers)} (Dense: 128 neurons, Dense: 64 neurons, Dense: 10 neurons)"
class_dist = ", ".join([f"Class {i}: {np.sum(y_test == i)}" for i in range(10)])

# Zapis do bazy
test = session.query(Test).filter_by(issue_key=ISSUE_KEY).first()
if not test:
    test = Test(
        issue_key=ISSUE_KEY,
        test_name="WYM-002 Robustness 01",
        test_type="Robustness",
        requirement="WYM-002",
        test_number="01"
    )
    session.add(test)
    session.commit()
    print(f"Added test record with issue_key: {ISSUE_KEY}")
else:
    print(f"Test record already exists for issue_key: {ISSUE_KEY}")

# Zapis wyniku dla danych oryginalnych
result_clean = TestResult(
    test_id=test.id,
    accuracy=test_accuracy,
    execution_time=execution_time,
    model_details=model_details,
    class_distribution=class_dist,
    status="Passed" if test_accuracy >= THRESHOLD else "Failed"
)
session.add(result_clean)
session.commit()
print(f"Added test result (clean) with accuracy: {test_accuracy}")

prediction_clean = Prediction(
    test_result_id=result_clean.id,
    y_true=json.dumps(y_test.tolist()),
    y_pred=json.dumps(y_pred_clean.tolist())
)
session.add(prediction_clean)
session.commit()
print(f"Added prediction record (clean) for test_result_id: {result_clean.id}")

cm_clean = confusion_matrix(y_test, y_pred_clean)
cm_json_clean = json.dumps(cm_clean.tolist())
confusion_entry_clean = ConfusionMatrix(test_result_id=result_clean.id, matrix_data=cm_json_clean)
session.add(confusion_entry_clean)
session.commit()
print(f"Added confusion matrix (clean) for test_result_id: {result_clean.id}")

# Zapis wyniku dla danych zaszumionych
result_noisy = TestResult(
    test_id=test.id,
    accuracy=test_accuracy_noisy,
    execution_time=execution_time,
    model_details=model_details,
    class_distribution=class_dist,
    status="Passed" if test_accuracy_noisy >= THRESHOLD else "Failed"
)
session.add(result_noisy)
session.commit()
print(f"Added test result (noisy) with accuracy: {test_accuracy_noisy}")

prediction_noisy = Prediction(
    test_result_id=result_noisy.id,
    y_true=json.dumps(y_test.tolist()),
    y_pred=json.dumps(y_pred_noisy.tolist())
)
session.add(prediction_noisy)
session.commit()
print(f"Added prediction record (noisy) for test_result_id: {result_noisy.id}")

cm_noisy = confusion_matrix(y_test, y_pred_noisy)
cm_json_noisy = json.dumps(cm_noisy.tolist())
confusion_entry_noisy = ConfusionMatrix(test_result_id=result_noisy.id, matrix_data=cm_json_noisy)
session.add(confusion_entry_noisy)
session.commit()
print(f"Added confusion matrix (noisy) for test_result_id: {result_noisy.id}")

# Diagnostyka: Sprawdzenie zapisanych danych
tests_count = session.query(Test).count()
results_count = session.query(TestResult).count()
predictions_count = session.query(Prediction).count()
confusion_matrices_count = session.query(ConfusionMatrix).count()
print(f"Total records in database:")
print(f"- Tests: {tests_count}")
print(f"- Test Results: {results_count}")
print(f"- Predictions: {predictions_count}")
print(f"- Confusion Matrices: {confusion_matrices_count}")

# Raport do JIRA dla danych zaszumionych (Robustness)
test_type_info = get_test_type_info("Odporności")
if test_accuracy_noisy >= THRESHOLD:
    report_success_to_jira(ISSUE_KEY, test_accuracy_noisy, "Robustness", test_type_info)
else:
    defect_key, precision_noisy, recall_noisy, f1_noisy, training_details, environment, top_misclassification_noisy = report_defect_to_jira(
        result_noisy.id, test_accuracy_noisy, THRESHOLD, ISSUE_KEY, y_test, y_pred_noisy, model, EPOCHS, TEST_SIZE, execution_time, model_details, class_dist
    )
    if defect_key:
        defect = Defect(
            test_result_id=result_noisy.id,
            issue_key=defect_key,
            description=f"Robustness Test: Accuracy below threshold: {test_accuracy_noisy:.4f}",
            precision=precision_noisy,
            recall=recall_noisy,
            f1_score=f1_noisy,
            training_details=training_details,
            environment=environment,
            top_misclassification=top_misclassification_noisy
        )
        session.add(defect)
        session.commit()
        print(f"Added defect record for test_result_id: {result_noisy.id}")

session.close()