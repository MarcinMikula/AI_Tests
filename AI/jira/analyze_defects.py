import json
import numpy as np
import re
from collections import Counter


# Funkcja do wyciągania danych z opisu defektu
def extract_metrics(description):
    metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "execution_time": None,
        "top_misclassification": None,
        "class_distribution": None
    }

    # Przetwarzanie opisu w formacie ADF
    for paragraph in description["content"]:
        text = paragraph["content"][0]["text"]

        # Accuracy
        if "Accuracy below threshold" in text:
            match = re.search(r"Accuracy below threshold: (\d+\.\d+)", text)
            if match:
                metrics["accuracy"] = float(match.group(1))

        # Metrics (Precision, Recall, F1-Score)
        if "Metrics: Precision" in text:
            match = re.search(r"Precision=(\d+\.\d+), Recall=(\d+\.\d+), F1-Score=(\d+\.\d+)", text)
            if match:
                metrics["precision"] = float(match.group(1))
                metrics["recall"] = float(match.group(2))
                metrics["f1_score"] = float(match.group(3))

        # Execution Time
        if "Execution Time" in text:
            match = re.search(r"Execution Time: (\d+\.\d+) seconds", text)
            if match:
                metrics["execution_time"] = float(match.group(1))

        # Top Misclassification
        if "Top Misclassification" in text:
            match = re.search(r"Class (\d+) misclassified as (\d+): (\d+) times", text)
            if match:
                metrics["top_misclassification"] = {
                    "true_class": int(match.group(1)),
                    "predicted_class": int(match.group(2)),
                    "count": int(match.group(3))
                }

        # Class Distribution
        if "Class Distribution" in text:
            class_counts = []
            matches = re.findall(r"Class \d+: \d+", text)
            for match in matches:
                count = int(re.search(r"Class \d+: (\d+)", match).group(1))
                class_counts.append(count)
            metrics["class_distribution"] = class_counts

    return metrics


# Wczytanie danych z JSON-ów
with open("my_defects.json", "r", encoding="utf-8") as f:
    my_defects = json.load(f)

with open("unassigned_defects.json", "r", encoding="utf-8") as f:
    unassigned_defects = json.load(f)

all_defects = my_defects + unassigned_defects
print(f"Total defects analyzed: {len(all_defects)}")

# Wyciąganie danych z każdego defektu
defect_metrics = []
for defect in all_defects:
    description = defect["fields"]["description"]
    metrics = extract_metrics(description)
    defect_metrics.append(metrics)

# Analiza 1: Średnie i statystyki metryk
accuracies = [m["accuracy"] for m in defect_metrics if m["accuracy"] is not None]
precisions = [m["precision"] for m in defect_metrics if m["precision"] is not None]
recalls = [m["recall"] for m in defect_metrics if m["recall"] is not None]
f1_scores = [m["f1_score"] for m in defect_metrics if m["f1_score"] is not None]
execution_times = [m["execution_time"] for m in defect_metrics if m["execution_time"] is not None]

print("\n=== Średnie metryki ===")
print(f"Średnia accuracy: {np.mean(accuracies):.4f} (std: {np.std(accuracies):.4f})")
print(f"Średnia precision: {np.mean(precisions):.4f} (std: {np.std(precisions):.4f})")
print(f"Średnia recall: {np.mean(recalls):.4f} (std: {np.std(recalls):.4f})")
print(f"Średnia F1-Score: {np.mean(f1_scores):.4f} (std: {np.std(f1_scores):.4f})")
print(
    f"Średni czas wykonania: {np.mean(execution_times):.2f} s (min: {np.min(execution_times):.2f}, max: {np.max(execution_times):.2f})")

# Analiza 2: Najczęstsze pomyłki w klasyfikacji
misclassifications = []
for m in defect_metrics:
    if m["top_misclassification"]:
        true_class = m["top_misclassification"]["true_class"]
        pred_class = m["top_misclassification"]["predicted_class"]
        count = m["top_misclassification"]["count"]
        misclassifications.append((true_class, pred_class, count))

# Sumowanie pomyłek dla każdej pary (true_class, pred_class)
misclass_counter = Counter([(true, pred) for true, pred, count in misclassifications for _ in range(count)])
print("\n=== Najczęstsze pomyłki ===")
for (true_class, pred_class), count in misclass_counter.most_common(5):
    print(f"Class {true_class} mylona z {pred_class}: {count} razy")

# Analiza 3: Rozkład klas w danych testowych
all_class_distributions = [m["class_distribution"] for m in defect_metrics if m["class_distribution"]]
if all_class_distributions:
    # Zakładamy, że rozkład klas jest taki sam w każdym defekcie
    class_distribution = np.array(all_class_distributions[0])
    print("\n=== Rozkład klas w danych testowych ===")
    print(f"Średnia liczba próbek na klasę: {np.mean(class_distribution):.2f} (std: {np.std(class_distribution):.2f})")
    print(f"Min liczba próbek: {np.min(class_distribution)}, Max: {np.max(class_distribution)}")
    print(f"Rozkład: {class_distribution}")