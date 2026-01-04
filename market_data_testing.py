import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("nse_all_stock_data(1).csv")

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date').reset_index(drop=True)

df = df.drop_duplicates(subset='Date')

df = df.fillna(method='ffill')

df['daily_return'] = df['Close'].pct_change()

df['rolling_volatility'] = df['daily_return'].rolling(window=20).std()

df['ma_20'] = df['Close'].rolling(window=20).mean()

df['signal'] = 0
df.loc[df['Close'] > df['ma_20'], 'signal'] = 1

df['signal'] = df['signal'].shift(1)

df['strategy_return'] = df['signal'] * df['daily_return']

df = df.fillna(0)

df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

df['cumulative_max'] = df['cumulative_return'].cummax()

df['drawdown'] = df['cumulative_return'] / df['cumulative_max'] - 1

max_drawdown = df['drawdown'].min()

sharpe_ratio = (
    df['strategy_return'].mean() /
    df['strategy_return'].std()
) * np.sqrt(252)

print("----- STRATEGY PERFORMANCE -----")
print(f"Final Cumulative Return: {df['cumulative_return'].iloc[-1]:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['cumulative_return'])
plt.title("Equity Curve (Cumulative P&L)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df['Date'], df['drawdown'])
plt.title("Drawdown Curve")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.show()

df.to_csv("backtest_results.csv", index=False)

print("\nBacktest completed successfully.")
print("Results saved to backtest_results.csv")
