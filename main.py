import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 1. QUANTITATIVE FINANCE: PORTFOLIO OPTIMIZATION
# ==========================================
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'GLD']
print(f"Fetching data for tickers: {tickers}")

# Download 1 year of historical adjusted closing prices
data = yf.download(tickers, period="1y", interval="1d", auto_adjust=True)
prices = data['Close'] if 'Close' in data else data

# Calculate Daily Returns & Annualized Metrics
returns = prices.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Portfolio Performance Functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    ret, std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / std

# Optimization Constraints & Bounds
num_assets = len(tickers)
args = (mean_returns, cov_matrix)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = num_assets * [1.0 / num_assets,]

# Solve Optimization Model
opts = minimize(negative_sharpe, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opts.x
opt_ret, opt_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
opt_sharpe = (opt_ret - 0.04) / opt_std

# Plot Allocation Chart
plt.figure(figsize=(8, 4))
plt.bar(tickers, optimal_weights * 100, color='skyblue', edgecolor='navy')
plt.title('Optimal Markowitz Portfolio Weights (%)')
plt.ylabel('Weight (%)')
plt.tight_layout()
plt.savefig('portfolio_weights.png')
plt.close()

# ==========================================
# 2. MANAGEMENT SCIENCE: SUPPLY CHAIN NETWORK LP MODEL
# ==========================================
# Network Problem: Minimize shipping costs from 2 Factories to 3 Warehouses
supply = [1000, 1500]       # Plant capacity
demand = [700, 800, 900]    # Warehouse demand
# Shipping cost matrix ($/unit)
costs = np.array([
    [4, 6, 8],  # Plant A to W1, W2, W3
    [5, 3, 7]   # Plant B to W1, W2, W3
])

# Solve using scipy.optimize.linprog
c = costs.flatten()
A_eq = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1]
])
b_eq = [supply[0], supply[1], demand[0], demand[1], demand[2]]

res = minimize(lambda x: np.dot(x, c), x0=np.zeros(6), bounds=[(0, None)]*6,
               constraints={'type': 'eq', 'fun': lambda x: np.dot(A_eq, x) - b_eq})
min_cost = res.fun

# ==========================================
# 3. GENERATE DYNAMIC PUBLIC README.MD
# ==========================================
timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

readme_content = f"""# Quantitative Studies & Management Science Analytics Pipeline

*Automated model execution timestamp:* **{timestamp}**

## 📊 1. Quantitative Portfolio Optimization (Markowitz Model)
- **Target Assets:** `{", ".join(tickers)}`
- **Expected Annual Return:** `{opt_ret * 100:.2f}%`
- **Annualized Volatility:** `{opt_std * 100:.2f}%`
- **Sharpe Ratio (Risk-Free Rate 4%):** `{opt_sharpe:.2f}`

### Optimal Asset Allocation
![Portfolio Weights](portfolio_weights.png)

| Ticker | Optimal Weight |
|--------|----------------|
"""
for t, w in zip(tickers, optimal_weights):
    readme_content += f"| **{t}** | `{w * 100:.2f}%` |\n"

readme_content += f"""
---

## 🏭 2. Management Science: Supply Chain Network Optimization
- **Problem Statement:** Minimum Cost Flow Linear Program
- **Calculated Minimum Logistics Cost:** `${min_cost:,.2f}`

---
*Powered by Python (`scipy`, `yfinance`) and GitHub Actions.*
"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("Analytics pipeline executed successfully.")
