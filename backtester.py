import pandas as pd

class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []

    def run(self, df, signals, stop_levels, scaler):
        position = 0
        for i in range(1, len(df)):
            if signals[i] != 0 and position == 0:
                vol = df['close'].rolling(10).std().iloc[i]
                draw = 1 - (self.equity / self.initial_capital)
                position = scaler.compute_position_size(self.equity, vol, draw) * signals[i]
            elif position != 0 and df['close'].iloc[i] < stop_levels.iloc[i]:
                self.equity += position * (df['close'].iloc[i] - df['close'].iloc[i-1])
                position = 0
            self.equity_curve.append(self.equity)
        return self.equity_curve
