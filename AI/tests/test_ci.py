import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import confusion_matrix
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import time
import json
from AI.jira.jira_utils import report_defect_to_jira, report_success_to_jira
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

# Test CI
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

# Test na danych oryginalnych
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(x_test), axis=1)

THRESHOLD = 0.95  # Próg dla CI Test – oczekujemy accuracy >= 95%
TEST_SIZE = len(x_test)

model_details = f"Layers={len(model.layers)} (Dense: 128 neurons, Dense: 64 neurons, Dense: 10 neurons)"
class_dist = ", ".join([f"Class {i}: {np.sum(y_test == i)}" for i in range(10)])

# Zapis do bazy
test = session.query(Test).filter_by(issue_key=ISSUE_KEY).first()
if not test:
    test = Test(
        issue_key=ISSUE_KEY,
        test_name="WYM-008 CI 01",
        test_type="CI",
        requirement="WYM-008",
        test_number="01"
    )
    session.add(test)
    session.commit()

result = TestResult(
    test_id=test.id,
    accuracy=test_accuracy,
    execution_time=execution_time,
    model_details=model_details,
    class_distribution=class_dist,
    status="Passed" if test_accuracy >= THRESHOLD else "Failed"
)
session.add(result)
session.commit()

prediction = Prediction(
    test_result_id=result.id,
    y_true=json.dumps(y_test.tolist()),
    y_pred=json.dumps(y_pred.tolist())
)
session.add(prediction)
session.commit()

cm = confusion_matrix(y_test, y_pred)
cm_json = json.dumps(cm.tolist())
confusion_entry = ConfusionMatrix(test_result_id=result.id, matrix_data=cm_json)
session.add(confusion_entry)
session.commit()

# Raport do JIRA
test_type_info = get_test_type_info("CI")
if test_accuracy >= THRESHOLD:
    report_success_to_jira(ISSUE_KEY, test_accuracy, "CI", test_type_info)
else:
    defect_key, precision, recall, f1, training_details, environment, top_misclassification = report_defect_to_jira(
        result.id, test_accuracy, THRESHOLD, ISSUE_KEY, y_test, y_pred, model, EPOCHS, TEST_SIZE, execution_time, model_details, class_dist
    )
    if defect_key:
        defect = Defect(
            test_result_id=result.id,
            issue_key=defect_key,
            description=f"CI Test: Accuracy below threshold: {test_accuracy:.4f}",
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_details=training_details,
            environment=environment,
            top_misclassification=top_misclassification
        )
        session.add(defect)
        session.commit()

session.close()