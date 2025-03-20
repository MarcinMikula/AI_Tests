import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import platform
import tensorflow as tf
import numpy as np
from Algorytmy.config import JIRA_URL, API_TOKEN, EMAIL
from datetime import datetime

# JIRA config
headers = {"Accept": "application/json", "Content-Type": "application/json"}
auth = (EMAIL, API_TOKEN)


# Funkcja sprawdzająca typ zgłoszenia
def get_issue_type(issue_key):
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}"
    response = requests.get(url, headers=headers, auth=auth)

    if response.status_code == 200:
        data = response.json()
        issue_type = data["fields"]["issuetype"]["name"]
        print(f"Issue {issue_key} type: {issue_type}")  # Diagnostyka
        return issue_type
    else:
        print(f"Failed to fetch issue {issue_key}. Status code: {response.status_code}")
        print(response.text)
        return None


def report_defect_to_jira(test_result_id, accuracy, threshold, parent_issue_key, y_true, y_pred, model, epochs,
                          test_size, execution_time, model_details, class_dist):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    tf_version = tf.__version__
    python_version = platform.python_version()
    os_version = platform.system() + " " + platform.release()

    cm = confusion_matrix(y_true, y_pred)
    # Dynamiczne określenie liczby klas na podstawie rozmiaru macierzy pomyłek
    num_classes = cm.shape[0]
    errors = [(i, j, cm[i, j]) for i in range(num_classes) for j in range(num_classes) if i != j and cm[i, j] > 0]
    top_error = max(errors, key=lambda x: x[2], default=(None, None, 0))
    top_misclassification = f"Class {top_error[0]} misclassified as {top_error[1]}: {top_error[2]} times" if top_error[
                                                                                                                 0] is not None else "No significant errors"

    training_details = f"Optimizer=adam, LR=0.001, Batch Size=32, Loss=sparse_categorical_crossentropy, Epochs={epochs}"
    environment = f"TensorFlow v{tf_version}, Python {python_version}, {os_version}"

    # Przygotowanie wyniku testu (do dodania jako komentarz)
    current_result = {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": f"Retest on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}:\n"
                                f"Accuracy: {accuracy:.4f} (Threshold: {threshold})\n"
                                f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}\n"
                                f"Execution Time: {execution_time:.2f} seconds\n"
                                f"Top Misclassification: {top_misclassification}"
                    }
                ]
            }
        ]
    }

    # Sprawdzenie typu zgłoszenia parent_issue_key
    issue_type = get_issue_type(parent_issue_key)
    if issue_type is None:
        print("Failed to determine issue type. Aborting.")
        return None, None, None, None, None, None, None

    if issue_type in ["Bug", "Błąd w programie"]:
        # parent_issue_key to defekt – aktualizujemy podsumowanie i dodajemy komentarz
        defect_key = parent_issue_key

        # Aktualizacja podsumowania
        update_payload = {
            "fields": {
                "summary": f"Test failed: Accuracy {accuracy:.4f} < {threshold}",
                "labels": ["AI", "TestFailure"]
            }
        }
        url = f"{JIRA_URL}/rest/api/3/issue/{defect_key}"
        response = requests.put(url, headers=headers, auth=auth, data=json.dumps(update_payload))
        if response.status_code != 204:
            print(f"Failed to update defect {defect_key}. Status code: {response.status_code}")
            print(response.text)
            return None, None, None, None, None, None, None

        # Dodanie nowego wyniku jako komentarz
        comment_url = f"{JIRA_URL}/rest/api/3/issue/{defect_key}/comment"
        comment_payload = {
            "body": current_result
        }
        comment_response = requests.post(comment_url, headers=headers, auth=auth, data=json.dumps(comment_payload))
        if comment_response.status_code == 201:
            print(f"Added retest comment to defect {defect_key}")
        else:
            print(f"Failed to add comment to defect {defect_key}. Status code: {comment_response.status_code}")
            print(comment_response.text)
    else:
        # parent_issue_key to nie defekt – tworzymy nowy
        defect_description = {
            "version": 1,
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [
                    {"type": "text", "text": f"Accuracy below threshold: {accuracy:.4f} (Threshold: {threshold})"}]},
                {"type": "paragraph", "content": [{"type": "text",
                                                   "text": f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": f"Model Details: {model_details}"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": f"Training: {training_details}"}]},
                {"type": "paragraph", "content": [
                    {"type": "text", "text": f"Test Data: Size={test_size} samples, Class Distribution={class_dist}"}]},
                {"type": "paragraph",
                 "content": [{"type": "text", "text": f"Execution Time: {execution_time:.2f} seconds"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": f"Environment: {environment}"}]},
                {"type": "paragraph",
                 "content": [{"type": "text", "text": f"Top Misclassification: {top_misclassification}"}]}
            ]
        }

        defect_payload = {
            "fields": {
                "project": {"key": "SCRUM"},
                "summary": f"Test failed: Accuracy {accuracy:.4f} < {threshold}",
                "description": defect_description,
                "issuetype": {"name": "Bug"},
                "labels": ["AI", "TestFailure"]
            }
        }
        url = f"{JIRA_URL}/rest/api/3/issue"
        response = requests.post(url, headers=headers, auth=auth, data=json.dumps(defect_payload))
        if response.status_code != 201:
            print(f"Failed to report defect. Status code: {response.status_code}")
            print(response.text)
            return None, None, None, None, None, None, None

        defect_key = response.json()["key"]
        print(f"Defect reported: {defect_key}")

        # Linkowanie defektu do parent_issue_key
        link_payload = {"type": {"name": "Blocks"}, "inwardIssue": {"key": parent_issue_key},
                        "outwardIssue": {"key": defect_key}}
        link_url = f"{JIRA_URL}/rest/api/3/issueLink"
        link_response = requests.post(link_url, headers=headers, auth=auth, data=json.dumps(link_payload))
        if link_response.status_code == 201:
            print(f"Defect {defect_key} linked to {parent_issue_key}")

    # Generowanie i załączanie confusion matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f"{cm_percent[i, j]:.1f}%", ha="center", va="center", color="red", fontsize=8)
    plt.title(f"Confusion Matrix - Accuracy: {accuracy:.4f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.close()

    attach_url = f"{JIRA_URL}/rest/api/3/issue/{defect_key}/attachments"
    attach_headers = {"X-Atlassian-Token": "no-check"}
    with open("confusion_matrix.png", "rb") as file:
        files = {"file": ("confusion_matrix.png", file)}
        attach_response = requests.post(attach_url, headers=attach_headers, auth=auth, files=files)
        if attach_response.status_code == 200:
            print(f"Confusion matrix attached to {defect_key}")
        else:
            print(f"Failed to attach confusion matrix. Status code: {attach_response.status_code}")
            print(attach_response.text)

    return defect_key, precision, recall, f1, training_details, environment, top_misclassification


def report_success_to_jira(issue_key, accuracy, test_type, test_type_info=None):
    description_content = [
        {"type": "paragraph", "content": [{"type": "text", "text": f"Test accuracy: {accuracy:.4f}"}]}
    ]
    if test_type_info:
        description_content.extend([
            {"type": "paragraph", "content": [{"type": "text", "text": f"Cel testu: {test_type_info['goal']}"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": f"Przykład: {test_type_info['example']}"}]},
            {"type": "paragraph",
             "content": [{"type": "text", "text": f"Dane testowe: {test_type_info['test_data']}"}]},
            {"type": "paragraph",
             "content": [{"type": "text", "text": f"Wykorzystane narzędzia: {test_type_info['tools_used']}"}]},
            {"type": "paragraph",
             "content": [{"type": "text", "text": f"Manipulacja danymi: {test_type_info['data_manipulation']}"}]}
        ])

    payload = {
        "fields": {
            "summary": f"{test_type} Test - Accuracy {accuracy:.4f}",
            "labels": ["AI", "TestPassed"],
            "description": {
                "version": 1,
                "type": "doc",
                "content": description_content
            }
        },
        "update": {
            "comment": [
                {
                    "add": {
                        "body": {
                            "version": 1,
                            "type": "doc",
                            "content": [
                                {"type": "paragraph",
                                 "content": [{"type": "text", "text": f"{test_type} Test accuracy: {accuracy:.4f}"}]}
                            ]
                        }
                    }
                }
            ]
        }
    }
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}"
    response = requests.put(url, headers=headers, auth=auth, data=json.dumps(payload))
    if response.status_code == 204:
        print(f"Successfully updated {issue_key}")
    else:
        print(f"Failed to update {issue_key}. Status code: {response.status_code}")
        print(response.text)