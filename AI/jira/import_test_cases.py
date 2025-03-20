import os
import requests
import json
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime

# JIRA config
JIRA_URL = "https://marcin00001a.atlassian.net"
API_TOKEN = "ATATT3xFfGF0z9-qBAGA_JIbha3zOj5goqGg94KXRv4UOTxzMIcLWcg26zdz_oSvoiX6tHmbaBZwLOMFLKnOu69T8UJIGYSc0LHs8PSay2zAccxpU5cdoleXCPSqon8GLk-GxpwbPvcmLIXADnAb3imxXeGwECiTB59WnwLIBdNeNIKQUvmvqNc=454AA3A1"  # Wygeneruj w ustawieniach JIRA
EMAIL = "marcin00001a@gmail.com"  # Twój email do logowania w JIRA
STORY_KEY = "SCRUM-13"  # Story, pod którą importujemy podzadania
headers = {"Accept": "application/json", "Content-Type": "application/json"}
auth = (EMAIL, API_TOKEN)

# Konfiguracja bazy danych (SQLite)
# Używamy tej samej bazy co w głównym skrypcie
engine = create_engine('sqlite:///test_results.db', echo=True)
Base = declarative_base()


# Definicja tabeli TESTS (tylko ta potrzebna do importu)
class Test(Base):
    __tablename__ = 'tests'
    id = Column(Integer, primary_key=True)
    issue_key = Column(String, unique=True, nullable=False)
    test_name = Column(String, nullable=False)
    test_type = Column(String, nullable=False)
    requirement = Column(String, nullable=False)
    test_number = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Tworzenie tabeli (jeśli nie istnieje)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# Funkcja importu testów z CSV do bazy i JIRA
def import_test_cases(csv_file, story_key):
    # Odczyt CSV
    df = pd.read_csv(csv_file)
    print("Imported test cases from CSV:")
    print(df)

    for index, row in df.iterrows():
        # Generowanie test_name i tymczasowego issue_key
        test_name = f"{row['requirement']} {row['test_type']} {row['test_number']}"
        temp_issue_key = f"TEMP-{index}"

        # Sprawdzenie, czy test już istnieje w bazie
        existing_test = session.query(Test).filter_by(test_name=test_name).first()
        if existing_test:
            print(f"Test {test_name} already exists in database with issue_key: {existing_test.issue_key}")
            continue

        # Zapis do bazy
        test = Test(
            issue_key=temp_issue_key,
            test_name=test_name,
            test_type=row['test_type'],
            requirement=row['requirement'],
            test_number=row['test_number']
        )
        session.add(test)
        session.commit()

        # Tworzenie podzadania w JIRA
        payload = {
            "fields": {
                "project": {"key": "SCRUM"},
                "parent": {"key": story_key},  # Powiązanie z SCRUM-13
                "summary": test_name,
                "description": {
                    "version": 1,
                    "type": "doc",
                    "content": [
                        {"type": "paragraph", "content": [{"type": "text",
                                                           "text": f"Test case: {test_name}, Type: {row['test_type']}, Requirement: {row['requirement']}"}]}
                    ]
                },
                "issuetype": {"name": "Podzadanie"}
            }
        }
        url = f"{JIRA_URL}/rest/api/3/issue"
        response = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload))
        if response.status_code == 201:
            jira_issue_key = response.json()["key"]
            print(f"Created sub-task in JIRA: {jira_issue_key}")

            # Aktualizacja issue_key w bazie
            test.issue_key = jira_issue_key
            session.commit()
        else:
            print(f"Failed to create sub-task. Status code: {response.status_code}")
            print(response.text)


# Wykonanie importu
import_test_cases("../test_cases.csv", STORY_KEY)

# Zamknięcie sesji
session.close()