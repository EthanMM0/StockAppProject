The Code will create 2 json files, these files will be state.json and price_data.json.

the state.json will be holding the trend data for the hgih/low trend values and the file will also contain the values for prediction

The price_data.json holds every updated value during the runtime of the application and cleans it every 20 minutes to avoid bad predictions

The code essentially using the price_data.json to get the values of highs and lows based off my trading strategy to effectively find highs, lows, and resistance as well as using the last 30 values 
to predict the next price over the next 1 minute which updates every 20 seconds 

the state.json essentially holds the state of the vlaues for the tkinter application for high, low, etc
