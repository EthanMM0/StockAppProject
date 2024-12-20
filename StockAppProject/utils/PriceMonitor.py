import json
from utils.utility import Utility

print(dir(Utility))


class PriceMonitor:
    """ Threshold for calculating Supply/Demand | Increase/Decrease of $0.30 within 10 seconds"""
    def __init__(self, threshold=0.30, time_window=10): 
        self.threshold = threshold
        self.time_window = time_window
        self.prices = []
        self.current_high = float('-inf')
        self.current_low = float('inf')
        self.prev_high = None
        self.prev_low = None

    def record_price(self, price):
        """ Record the latest price """
        self.prices.append(price)
        if len(self.prices) > self.time_window:
            self.prices.pop(0)  # Maintain the time window by popping the oldest price

        # Update high/low tracking
        self.update_price(price)

    def check_zone(self):
        """ Check if we are in a supply or demand zone based on recorded prices """
        if len(self.prices) < 2:
            return "None"

        # Load the price data from JSON to evaluate zones
        saved_data = Utility.load_data_from_json("price_data.json")  # change price_data.json to folder you want it to be in EX: ./folder/folder/price_data.json
        if saved_data:
            self.prices = [entry["price"] for entry in saved_data]  # Update prices from JSON file

        min_price = min(self.prices)
        max_price = max(self.prices)

        if max_price - min_price >= self.threshold:
            return "Supply Zone | Sell"
        elif min_price - max_price <= -self.threshold:
            return "Demand Zone | Buy"
        else:
            return "Neutral Zone | Slow"

    def clear_prices(self):
        """ Clear recorded prices """
        self.prices.clear()
        self.current_high = float('-inf')
        self.current_low = float('inf')
        self.prev_high = None
        self.prev_low = None

    def update_price(self, new_price):
        # Update current high/low
        if new_price > self.current_high:
            self.current_high = new_price
        if new_price < self.current_low:
            self.current_low = new_price

        # Update previous high/low values after each update cycle
        self.prev_high = self.current_high
        self.prev_low = self.current_low

    def get_current_values(self):
        return {
            'current_high': self.current_high,
            'current_low': self.current_low,
            'prev_high': self.prev_high,
            'prev_low': self.prev_low
        }