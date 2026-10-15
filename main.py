import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, linprog
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
# Objective: Minimize shipping costs from 2 Factories to 3 Warehouses
# Shipping costs ($/unit):
# Plant A -> W1: $4, W2: $6, W3: $8
# Plant B -> W1: $5, W2: $3, W3: $7
c = [4, 6, 8, 5, 3, 7]

# Supply constraints (Capacity: Plant A <= 1000, Plant B <= 1500)
A_ub = [
    [1, 1, 1, 0, 0, 0],  # Plant A capacity limit
    [0, 0, 0, 1, 1, 1]   # Plant B capacity limit
]
b_ub = [1000, 1500]

# Demand constraints (W1 = 700, W2 = 800, W3 = 900)
A_eq = [
    [1, 0, 0, 1, 0, 0],  # Warehouse 1 demand
    [0, 1, 0, 0, 1, 0],  # Warehouse 2 demand
    [0, 0, 1, 0, 0, 1]   # Warehouse 3 demand
]
b_eq = [700, 800, 900]

# All shipments must be non-negative
bounds = [(0, None)] * 6

# Solve using Simplex Linear Programming solver
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
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
