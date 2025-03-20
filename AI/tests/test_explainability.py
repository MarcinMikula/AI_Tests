import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import matplotlib.pyplot as plt
from AI.jira.jira_utils import report_success_to_jira
from Algorytmy.config import JIRA_URL
import requests
import time
from Algorytmy.config import ISSUE_KEY, DB_PATH


# Konfiguracja bazy danych (SQLite)
print(f"Using database path: {DB_PATH}")
engine = create_engine(f'sqlite:///{DB_PATH}', echo=True)
Base = declarative_base()

# Definicja tabel (takie same jak w test_robustness.py)
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

# Wczytanie danych MNIST
tf.random.set_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Tworzenie modelu (taki sam jak w test_robustness.py)
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

# Wybór jednego obrazu testowego (pierwszy obraz z x_test)
image = x_test[0:1]  # Kształt: (1, 784)
true_label = y_test[0]  # Prawdziwa etykieta

# Predykcja dla wybranego obrazu
prediction = model.predict(image)
predicted_label = np.argmax(prediction, axis=1)[0]

# Obliczenie saliency map (gradienty predykcji względem pikseli)
image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(image_tensor)
    prediction = model(image_tensor)
    predicted_class_output = prediction[0, predicted_label]  # Wynik dla przewidzianej klasy
gradients = tape.gradient(predicted_class_output, image_tensor)

# Przekształcenie gradientów do wartości absolutnych i normalizacja
saliency_map = np.abs(gradients.numpy())
saliency_map = saliency_map / np.max(saliency_map)  # Normalizacja do [0, 1]
saliency_map = saliency_map.reshape(28, 28)  # Powrót do kształtu 2D (28x28)

# Wizualizacja oryginalnego obrazu i saliency map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image.reshape(28, 28), cmap='gray')
ax1.set_title(f"Original Image (True: {true_label}, Pred: {predicted_label})")
ax1.axis('off')

ax2.imshow(saliency_map, cmap='hot')
ax2.set_title("Saliency Map (Important Pixels)")
ax2.axis('off')

plt.savefig("saliency_map.png")
plt.close()

# Zapis wyników do bazy danych
test = session.query(Test).filter_by(issue_key=ISSUE_KEY).first()
if not test:
    test = Test(
        issue_key=ISSUE_KEY,
        test_name="WYM-003 Explainability 01",
        test_type="Explainability",
        requirement="WYM-003",
        test_number="01"
    )
    session.add(test)
    session.commit()

model_details = f"Layers={len(model.layers)} (Dense: 128 neurons, Dense: 64 neurons, Dense: 10 neurons)"
class_dist = ", ".join([f"Class {i}: {np.sum(y_test == i)}" for i in range(10)])

result = TestResult(
    test_id=test.id,
    accuracy=float(prediction[0, predicted_label]),  # Prawdopodobieństwo przewidzianej klasy
    execution_time=execution_time,
    model_details=model_details,
    class_distribution=class_dist,
    status="Passed"  # Test Explainability nie ma progu, zakładamy sukces
)
session.add(result)
session.commit()

# Raport do JIRA
test_type_info = get_test_type_info("Explainability")
report_success_to_jira(ISSUE_KEY, float(prediction[0, predicted_label]), "Explainability", test_type_info)

# Załączanie saliency map do JIRA jako załącznik
from AI.jira.jira_utils import auth  # Import nagłówków i autoryzacji
attach_url = f"{JIRA_URL}/rest/api/3/issue/{ISSUE_KEY}/attachments"
attach_headers = {"X-Atlassian-Token": "no-check"}
with open("saliency_map.png", "rb") as file:
    files = {"file": ("saliency_map.png", file)}
    attach_response = requests.post(attach_url, headers=attach_headers, auth=auth, files=files)
    if attach_response.status_code == 200:
        print(f"Saliency map attached to {ISSUE_KEY}")
    else:
        print(f"Failed to attach saliency map. Status code: {attach_response.status_code}")
        print(attach_response.text)

session.close()