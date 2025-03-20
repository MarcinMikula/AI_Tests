import subprocess
import sys
import os

# Lista testów do uruchomienia
TESTS_TO_RUN = [
    "test_functional.py",
    "test_explainability.py",
    "test_robustness.py"
]

def run_test(test_file):
    print(f"\n=== Running test: {test_file} ===")
    try:
        # Uruchomienie skryptu testowego w osobnym procesie
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.path.dirname(os.path.abspath(__file__)),  # Uruchamiamy w folderze tests/
            capture_output=True,
            text=True
        )
        # Wyświetlenie wyniku
        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        if result.returncode != 0:
            print(f"Test {test_file} failed with exit code {result.returncode}")
        else:
            print(f"Test {test_file} completed successfully")
    except Exception as e:
        print(f"Failed to run test {test_file}: {str(e)}")

def main():
    print("Starting test suite...")
    for test_file in TESTS_TO_RUN:
        run_test(test_file)
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()