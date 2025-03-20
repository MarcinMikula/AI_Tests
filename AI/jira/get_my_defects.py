import requests
import json
from Algorytmy.config import JIRA_URL, API_TOKEN, EMAIL

# Konfiguracja JIRA
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}
auth = (EMAIL, API_TOKEN)


def get_my_defects():
    # JQL: Defekty przypisane do mnie
    jql = 'assignee = currentUser() AND issuetype = "Bug"'
    start_at = 0
    max_results = 50  # Maksymalna liczba wyników na żądanie
    all_defects = []

    while True:
        # URL z paginacją
        url = f"{JIRA_URL}/rest/api/3/search?jql={jql}&startAt={start_at}&maxResults={max_results}"
        response = requests.get(url, headers=headers, auth=auth)

        if response.status_code != 200:
            print(f"Failed to fetch defects. Status code: {response.status_code}")
            print(response.text)
            break

        data = response.json()
        issues = data.get("issues", [])
        all_defects.extend(issues)

        # Sprawdzanie paginacji
        start_at += len(issues)
        total = data.get("total", 0)
        print(f"Fetched {start_at}/{total} defects...")

        if start_at >= total:
            break

    return all_defects


def save_to_file(defects, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(defects, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(defects)} defects to {filename}")


if __name__ == "__main__":
    my_defects = get_my_defects()
    save_to_file(my_defects, "my_defects.json")