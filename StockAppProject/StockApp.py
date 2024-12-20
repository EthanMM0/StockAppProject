import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter, MinuteLocator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import yfinance as yf
import threading
import time
import json
import os
from utils.utility import Utility
from utils.PriceMonitor import PriceMonitor

# initializing pre stockapp
        


    
import pandas as pd
from datetime import datetime, timedelta
import random
from sklearn.linear_model import LinearRegression  # Added for prediction
import numpy as np

class StockApp: 
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Tracker with AI Trend Prediction")
        
        # Initialize the PriceMonitor for Supply/Demand Zone functionality
        self.price_monitor = PriceMonitor(threshold=0.50, time_window=10)
        
        self.last_30_lows = []  # Track last 30 lows before a rise
        self.last_30_highs = []  # Track last 30 highs before a drop
        
        # Other initializations

        self.predicted_price_var = tk.StringVar(value="Predicted Price: $0.00")
        
        # GUI Components
        self.search_var = tk.StringVar()
        self.current_price_var = tk.StringVar(value="Current Price: $0.00")
        self.current_high_var = tk.StringVar(value="Current High: $0.00")
        self.current_low_var = tk.StringVar(value="Current Low: $0.00")
        self.trend_var = tk.StringVar(value="Trend: None")
        self.resistance_var = tk.StringVar(value="Resistance: False")
        self.create_gui()

        # Chart Data
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Thread control
        self.running = False
        self.ticker = None
        self.update_thread = None
        self.console_thread = None
        self.resistance_thread = None  # Thread for resistance checking

        # High/Low variables for trend detection
        self.prev_high = None
        self.prev_low = None
        self.current_trend = None
    
        # Separate window for trading strategy information
        self.trading_window = None

        # Data for prediction
        self.prediction_model = LinearRegression()
        self.prediction_data = []
        self.latest_price = None
        self.price_movement_threshold = 0.05  # Defines a threshold for significant price movement (in dollars)
        
        self.threshold_var = tk.DoubleVar(value=0.30)  # Default threshold value
        
        
        
    def save_state(self):
        """Save the current state (high/low values and trend) to a JSON file."""
        state = {
            "prev_high": self.prev_high,
            "prev_low": self.prev_low,
            "current_trend": self.current_trend,
            "prediction_data": self.prediction_data  # Save prediction data
        }
        try:
            with open("state.json", "w") as f: # change state.json to folder you want it to be in
                json.dump(state, f)
            print("State saved successfully.")
        except Exception as e:
            print(f"Error saving state: {e}")
        

    def identify_highs_and_lows(self, data, window_size=5):
        """Identify highs and lows in the stock data."""
        highs = []
        lows = []

        for i in range(window_size, len(data) - window_size):
            window = data[i - window_size:i + window_size + 1]
            current_point = data[i]

            if current_point == max(window):
                highs.append((data.index[i], current_point))
            elif current_point == min(window):
                lows.append((data.index[i], current_point))

        return highs, lows

    def update_high_low(self, data):
        """Update high and low values dynamically, considering trends and edge cases."""
        max_price = data.max().iloc[0]
        min_price = data.min().iloc[0]

        # Initialize high/low on the first update or after resetting
        if self.prev_high is None or self.prev_low is None:
            self.prev_high = max_price
            self.prev_low = min_price
            self.current_trend = "Consolidation"
            self.trend_var.set(f"Trend: {self.current_trend}")
            return

        # Threshold for detecting trends
        threshold = 0.01  # 1% fluctuation tolerance for error handling of stock
        price_range = self.prev_high - self.prev_low

        # Ignore minor fluctuations
        if abs(max_price - self.prev_high) < threshold * price_range and abs(min_price - self.prev_low) < threshold * price_range:
            self.current_trend = "Consolidation"
        elif max_price > self.prev_high:
            self.prev_high = max_price
            self.current_trend = "Uptrend"
        elif min_price < self.prev_low:
            self.prev_low = min_price
            self.current_trend = "Downtrend"

        # Update GUI
        self.trend_var.set(f"Trend: {self.current_trend}")
        self.current_high_var.set(f"Current High: ${self.prev_high:.2f}")
        self.current_low_var.set(f"Current Low: ${self.prev_low:.2f}")

        # Save state to JSON
        self.save_state()

        # Log for debugging purposes
        print(f"Trend: {self.current_trend}, Prev High: ${self.prev_high:.2f}, Prev Low: ${self.prev_low:.2f}")


    def update_last_30_lows(self, price):
        """Update the list of the last 30 lows before a rise."""
        self.last_30_lows.append(price)
        if len(self.last_30_lows) > 30:
            self.last_30_lows.pop(0)

    def update_last_30_highs(self, price):
        """Update the list of the last 30 highs before a drop."""
        self.last_30_highs.append(price)
        if len(self.last_30_highs) > 30:
            self.last_30_highs.pop(0)

    def get_last_30_lows(self):
        """Retrieve the last 30 lows before a rise."""
        return self.last_30_lows

    def get_last_30_highs(self):
        """Retrieve the last 30 highs before a drop."""
        return self.last_30_highs


    def create_gui(self):
        """Create the GUI for the stock tracker."""
        # Search bar
        search_frame = ttk.Frame(self.root)
        search_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(search_frame, text="Enter Ticker:").pack(side=tk.LEFT, padx=5)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=5)

        # Buttons for Search and Predictions
        ttk.Button(search_frame, text="Search", command=self.start_tracking).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Analysts' Predictions", command=self.create_analyst_prediction_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Settings", command=self.open_settings_window).pack(side=tk.LEFT, padx=5)  # New Settings button

        # Current Price and Trend Display
        info_frame = ttk.Frame(self.root)  # New frame for organizing labels
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        price_label = ttk.Label(info_frame, textvariable=self.current_price_var, font=("Arial", 12))
        price_label.pack(side=tk.LEFT, padx=10)

        high_label = ttk.Label(info_frame, textvariable=self.current_high_var, font=("Arial", 12))
        high_label.pack(side=tk.LEFT, padx=10)

        low_label = ttk.Label(info_frame, textvariable=self.current_low_var, font=("Arial", 12))
        low_label.pack(side=tk.LEFT, padx=10)

        trend_label = ttk.Label(info_frame, textvariable=self.trend_var, font=("Arial", 12))
        trend_label.pack(side=tk.LEFT, padx=10)

        resistance_label = ttk.Label(info_frame, textvariable=self.resistance_var, font=("Arial", 12))
        resistance_label.pack(side=tk.LEFT, padx=10)

        predicted_label = ttk.Label(info_frame, textvariable=self.predicted_price_var, font=("Arial", 12))
        predicted_label.pack(side=tk.LEFT, padx=10)

        # Chart Frame
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Prediction Button
        action_frame = ttk.Frame(self.root)  # New frame for action buttons
        action_frame.pack(fill=tk.X, padx=10, pady=5)

        self.prediction_button = ttk.Button(action_frame, text="Predict Trend", command=self.predict_trend)
        self.prediction_button.pack(side=tk.LEFT, padx=5)

        
    def filter_significant_price_movements(self, data):
        """Filter out insignificant price movements based on a defined threshold."""
        filtered_data = []
        for i in range(1, len(data)):
            # Calculate the price difference between consecutive data points
            price_diff = abs(data[i]["price"] - data[i - 1]["price"])
            
            # Only keep data points where the price movement exceeds the threshold
            if price_diff >= self.price_movement_threshold:
                filtered_data.append(data[i])
    
        return filtered_data
        
    def update_predictions(self):
        """Predicts the next price and updates the prediction window."""
        # Load data from JSON file
        saved_data = Utility.load_data_from_json("price_data.json") # change price_data.json to folder you want it to be in EX: ./folder/folder/price_data.json
        if saved_data:
            # Filter out insignificant price movements
            filtered_data = self.filter_significant_price_movements(saved_data)
            self.prediction_data = [entry["price"] for entry in filtered_data]  # Extract prices from filtered data
        
        # Check if we have enough data for prediction
        if len(self.prediction_data) >= 2:
            try:
                # Prepare data for prediction
                x = np.array(range(len(self.prediction_data))).reshape(-1, 1)  # Time points (x-axis)
                y = np.array(self.prediction_data).reshape(-1, 1)  # Prices (y-axis)

                # Train the linear regression model
                self.prediction_model.fit(x, y)

                # Predict the next 5 values (next 5 time steps)
                future_indices = np.array(range(len(self.prediction_data), len(self.prediction_data) + 5)).reshape(-1, 1)
                predicted_values = self.prediction_model.predict(future_indices)

                # Update the label with the predicted price trend (for the next 5 time steps)
                prediction_text = "Predicted Next Prices:\n"
                for idx, predicted_value in enumerate(predicted_values):
                    prediction_text += f"Time {len(self.prediction_data) + idx}: ${predicted_value[0]:.2f}\n"
                    
                # Update last predicted price and timestamp
                self.last_predicted_value = predicted_value[0][0]
                
                
                # Update GUI component for chart (if applicable)
                self.predicted_price_var.set(f"Predicted Price (1 min): ${self.last_predicted_price:.2f}")

                # Update the prediction label
                if self.prediction_label:
                    self.prediction_label.config(text=prediction_text)

                # Update the trading window with the last predicted price
                self.update_trading_info()

            except Exception as e:
                if self.prediction_label:
                    self.prediction_label.config(text=f"Error: {e}")
        
        # Schedule the next update in 20 seconds
        if self.prediction_label and self.prediction_label.winfo_exists():
            self.prediction_label.after(2000, self.update_predictions)


    def run_predictions(self):
        """Continuously predict the next price and update every 20 seconds."""
        while self.running:
            # Load data from JSON file
            saved_data = Utility.load_data_from_json("price_data.json")
            if saved_data:
                # Filter out insignificant price movements
                filtered_data = self.filter_significant_price_movements(saved_data)
                self.prediction_data = [entry["price"] for entry in filtered_data]  # Extract prices from filtered data
            
            if len(self.prediction_data) >= 2:
                try:
                    # Prepare data for prediction
                    x = np.array(range(len(self.prediction_data))).reshape(-1, 1)  # Time points (x-axis)
                    y = np.array(self.prediction_data).reshape(-1, 1)  # Prices (y-axis)

                    # Train the linear regression model
                    self.prediction_model.fit(x, y)

                    # Predict the next price
                    next_index = len(self.prediction_data)
                    predicted_value = self.prediction_model.predict(np.array([[next_index]]))

                    # Update prediction display
                    if self.predicted_price_var:
                        self.predicted_price_var.set(f"Predicted Price (1 min): ${predicted_value[0][0]:.2f}")
                except Exception as e:
                    print(f"Error in prediction: {e}")
            
            # Wait for 20 seconds before updating the prediction again
            time.sleep(20)
            
    def start_tracking(self):
        """Validate ticker, initialize threads, and manage trading data."""
        ticker = self.search_var.get().upper()
        if not ticker.isalpha():  # Validate ticker input
            tk.messagebox.showerror("Error", "Invalid ticker symbol. Please enter a valid one.")
            return

        if ticker:
            # Delete files when switching tickers
            if os.path.exists("price_data.json"):
                os.remove("price_data.json")
            if os.path.exists("state.json"):
                os.remove("state.json")

            self.ticker = ticker
            if not self.running:
                self.running = True
                self.update_thread = threading.Thread(target=self.update_chart, daemon=True)
                self.resistance_thread = threading.Thread(target=self.check_resistance, daemon=True)
                self.prediction_thread = threading.Thread(target=self.run_predictions, daemon=True)
                self.cleanup_thread = threading.Thread(target=self.periodic_cleanup, daemon=True)  # Add cleanup thread
                self.update_thread.start()
                self.resistance_thread.start()
                self.prediction_thread.start()
                self.cleanup_thread.start()  # Start the cleanup thread

            # Use self.latest_price safely
            if self.latest_price is not None:
                self.prediction_data.append(self.latest_price)
                if len(self.prediction_data) > 30:
                    self.prediction_data.pop(0)
            else:
                print("Latest price is not yet available.")

            # Create and update the trading window only if it doesn't exist
            if self.trading_window is None or not self.trading_window.winfo_exists():
                self.trading_window = tk.Toplevel(self.root)
                self.trading_window.title("Trading Strategy Info")
                tk.Label(self.trading_window, text=f"Latest Predicted Price: None", font=("Arial", 12)).pack(pady=5)
                self.update_trading_info()

        
    def update_trading_info(self):
        """Update the trading strategy information in the second window."""
        if self.trading_window is not None and self.trading_window.winfo_exists():
            # Clear previous information
            for widget in self.trading_window.winfo_children():
                widget.destroy()

            # Display updated supply/demand zone information
            zone = self.price_monitor.check_zone()
            zone_label = tk.Label(self.trading_window, text="Trading Strategy Information", font=("Arial", 14, "bold"))
            zone_label.pack(pady=10)

            if zone:
                tk.Label(self.trading_window, text=f"Current Zone: {zone}", font=("Arial", 12)).pack(pady=5)
            else:
                tk.Label(self.trading_window, text="Currently in no specific zone.", font=("Arial", 12)).pack(pady=5)
                
            

            # Additional trading strategy info could go here
            tk.Label(self.trading_window, text="Use supply/demand zones for entry/exit points.", font=("Arial", 12)).pack(pady=5)
            tk.Label(self.trading_window, text="Look for price movements beyond $0.30 to detect significant changes.", font=("Arial", 12)).pack(pady=5)
            

                
    def console_updates(self):
        while self.running:
            # Example of updating the console with some message testing purpose
            print("Console updating...")
            time.sleep(1)

    def fetch_data(self):
        try:
            # Download data for the given ticker
            data = yf.download(self.ticker, period="1d", interval="1m", prepost=False)

            # Convert to UTC if data is not tz-aware
            if data.index.tzinfo is None:
                data.index = data.index.tz_localize('UTC')  # Localize to UTC

            return data['Close']
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        
    def initialize_from_saved_data(self):
        """Initialize high/low and prediction values using saved data."""
        saved_data = Utility.load_data_from_json("price_data.json")
        if saved_data:
            # Recalculate prev_high and prev_low
            prices = [entry["price"] for entry in saved_data]
            self.prev_high = max(prices)
            self.prev_low = min(prices)
            self.prediction_data = prices  # Use the prices as prediction data
            print(f"Loaded Previous High: {self.prev_high}, Previous Low: {self.prev_low}")
            print(f"Loaded Prediction Data: {self.prediction_data}")
        else:
            self.prev_high = None
            self.prev_low = None
            self.prediction_data = []

    def update_chart(self):
        while self.running:
            data = self.fetch_data()
            if data is not None and not data.empty:  # Ensure data is valid and not empty
                self.ax.clear()

                # Get the current time and filter data up to now
                now = pd.Timestamp.now(tz='UTC')
                data = data[data.index <= now]  # Filter data to only include up to the current time

                # Extract the latest price correctly
                latest_price = data.values[-1] if isinstance(data.values[-1], (int, float)) else data.values[-1][0]
                self.latest_price = latest_price  # Store for prediction and GUI updates

                # Plot the data
                self.ax.plot(data.index, data.values, label=f"{self.ticker} Price")
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Price")
                self.ax.grid(True)  # Add grid for better visualization

                # Formatting x-axis for half-hour intervals
                self.ax.xaxis.set_major_locator(MinuteLocator(interval=30))
                self.ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
                self.ax.tick_params(axis="x", rotation=45)

                # Set x-axis limits to always show from 9:30 AM EST to 4 PM EST (14:30 UTC to 21:00 UTC)
                start_time = pd.Timestamp(now.date()) + pd.Timedelta(hours=14, minutes=30)  # 9:30 AM EST = 14:30 UTC
                end_time = pd.Timestamp(now.date()) + pd.Timedelta(hours=21)  # 4:00 PM EST = 21:00 UTC
                self.ax.set_xlim(start_time, end_time)

                # Display trend information dynamically (uptrend, downtrend, or consolidation)
                if self.current_trend == "Uptrend":
                    self.ax.set_title(f"Real-time Price for {self.ticker} (Uptrend)")
                elif self.current_trend == "Downtrend":
                    self.ax.set_title(f"Real-time Price for {self.ticker} (Downtrend)")
                else:
                    self.ax.set_title(f"Real-time Price for {self.ticker}")

                self.ax.legend()
                self.canvas.draw()

                # Save price and timestamp to a JSON file
                price_data = {"timestamp": str(now), "price": float(latest_price)}
                try:
                    with open("price_data.json", "r") as f:
                        all_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_data = []

                all_data.append(price_data)
                Utility.save_data_to_json("price_data.json", all_data)

                # Update GUI
                self.current_price_var.set(f"Current Price: ${latest_price:.2f}")

                # Store the data for trend prediction
                self.prediction_data.append(latest_price)
                if len(self.prediction_data) > 50:  # Limit data size for prediction
                    self.prediction_data.pop(0)

                # Update high/low values dynamically
                self.update_high_low(data)

                # Record price in price monitor for supply/demand zone check
                self.price_monitor.record_price(latest_price)

                # Update the demand and supply zone display
                self.update_trading_info()

            else:
                print(f"No data available for {self.ticker}. Retrying...")

            time.sleep(3)

            # Existing logic for handling a drop in price
            if self.last_30_highs is not None:
                self.update_last_30_highs(self.last_30_highs)  # Add the last high to the list
                self.last_30_highs = None

            # Existing logic for handling a rise in price
            if self.last_30_lows is not None:
                self.update_last_30_lows(self.last_30_lows)  # Add the last low to the list
                self.last_30_lows = None



    def check_resistance(self):
        while self.running:
            data = self.fetch_data()
            if data is not None:
                # Check if the price has stayed within a $0.50 range over the past minute
                now = pd.Timestamp.now(tz='UTC')
                one_minute_ago = now - pd.Timedelta(minutes=1)
                recent_data = data[data.index >= one_minute_ago]

                if not recent_data.empty:
                    max_price = recent_data.max()
                    min_price = recent_data.min()

                    # Ensure max_price and min_price are scalars (not Series)
                    max_price = max_price.iloc[0] if isinstance(max_price, pd.Series) else max_price
                    min_price = min_price.iloc[0] if isinstance(min_price, pd.Series) else min_price

                    # Now perform the comparison
                    if max_price - min_price <= 0.50:
                        self.resistance_var.set("Resistance: True")
                    else:
                        self.resistance_var.set("Resistance: False")

            time.sleep(15)  # Check every 15 seconds
            
    def reset_trend_data(self):
        # Reset trend-related variables when switching tickers
        self.prev_high = None
        self.prev_low = None
        self.current_trend = None
        self.trend_var.set("Trend: None")

    def predict_trend(self):
        """Manually predict the next price and display in a popup."""
        if len(self.prediction_data) < 2:
            tk.messagebox.showerror("Error", "Not enough data for prediction!")
            return

        try:
            # Prepare data for prediction
            x = np.array(range(len(self.prediction_data))).reshape(-1, 1)
            y = np.array(self.prediction_data).reshape(-1, 1)
        
            # Train the model
            self.prediction_model.fit(x, y)

            # Predict next value
            predicted_value = self.prediction_model.predict(np.array([[len(self.prediction_data)]]))

            # Display the prediction result
            tk.messagebox.showinfo("Prediction", f"Predicted Next Price: ${predicted_value[0][0]:.2f}")
        except Exception as e:
            tk.messagebox.showerror("Prediction Error", f"Error predicting price: {e}")
        
    def truncate_price_data(self, file, max_entries=50):
        """Keep only the last `max_entries` in the JSON file."""
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        # Keep only the last `max_entries`
        truncated_data = data[-max_entries:]
        
        # Save back to the file
        with open(file, "w") as f:
            json.dump(truncated_data, f)

    def periodic_cleanup(self):
        """Clear the price_data.json every 30 minutes, keeping only the last 30 entries."""
        while self.running:  # `app.running` ensures this runs only when the app is active
            self.truncate_price_data("price_data.json", max_entries=30)
            print("Price data truncated to last 30 entries.")
            time.sleep(1800)  # 30 minutes in seconds



    def create_analyst_prediction_window(self):
        """Creates a new window to display analysts' predictions."""
        if self.ticker is None:
            tk.messagebox.showerror("Error", "Please search for a ticker first!")
            return

        analyst_window = tk.Toplevel(self.root)
        analyst_window.title("Analysts' Predictions")

        try:
            stock = yf.Ticker(self.ticker)

            # Fetch analysts' recommendations
            recos = stock.recommendations
            if isinstance(recos, pd.DataFrame) and not recos.empty:
                latest_reco = recos.tail(1)
                tk.Label(analyst_window, text=f"Latest Recommendation:").pack(anchor="w", padx=10, pady=5)
                tk.Label(analyst_window, text=latest_reco.to_string(), justify="left").pack(anchor="w", padx=10, pady=5)
            else:
                tk.Label(analyst_window, text="No recommendations available.").pack(anchor="w", padx=10, pady=5)

            # Fetch Earnings Estimates
            earnings = stock.earnings
            if isinstance(earnings, pd.DataFrame) and not earnings.empty:
                tk.Label(analyst_window, text="Earnings Estimates:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
                estimates = earnings.to_string()
                tk.Label(analyst_window, text=estimates, justify="left").pack(anchor="w", padx=10, pady=5)
            else:
                tk.Label(analyst_window, text="No earnings estimates available.").pack(anchor="w", padx=10, pady=5)

            # earnings calendar
            calendar = stock.calendar
            if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                tk.Label(analyst_window, text="Earnings Calendar:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
                calendar_str = calendar.to_string()
                tk.Label(analyst_window, text=calendar_str, justify="left").pack(anchor="w", padx=10, pady=5)

        except Exception as e:
            tk.Label(analyst_window, text=f"Error fetching analysts' predictions: {e}").pack(anchor="w", padx=10, pady=5)
            
    def open_settings_window(self):
        """Open a settings window with a slider to adjust the threshold."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("350x350")

        tk.Label(settings_window, text="Threshold", font=("Arial", 14)).pack(pady=10)

        slider = ttk.Scale(
            settings_window, 
            from_=0.05, 
            to=1, 
            variable=self.threshold_var, 
            orient=tk.HORIZONTAL, 
            length=200
        )
        slider.pack(pady=10)

        tk.Label(settings_window, textvariable=self.threshold_var, font=("Arial", 12)).pack(pady=10)

        ttk.Button(
            settings_window, 
            text="Apply", 
            command=lambda: print(f"Threshold set to: {self.threshold_var.get()}")
        ).pack(pady=10)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()
