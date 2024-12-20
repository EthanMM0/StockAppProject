import json

class Utility:

    def load_data_from_json(file):
        """Load data with timestamps from a JSON file."""
        try:
            with open(file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_data_to_json(file, data):
        """Save data to a JSON file."""
        with open(file, "w") as f:
            json.dump(data, f)