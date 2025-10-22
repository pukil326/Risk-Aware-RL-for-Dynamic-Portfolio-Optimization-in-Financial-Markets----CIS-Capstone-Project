# Evaluation of Reinforcement Learning Methods vs. Traditional Technical Analysis in Quantitative Trading

##  Project Overview
This project aims to evaluate and compare the performance of several **Reinforcement Learning (RL)** methods with **Traditional Technical Analysis (TA)** strategies in quantitative trading.  
Using **S&P 500** data from the past **10 years**, we examine how RL agents make trading decisions based on either **raw price data** or a set of **20 technical indicators**.

The following RL algorithms will be implemented and tested:
- **Deep Q-Network (DQN)**
- **Proximal Policy Optimization (PPO)**
- **Actor–Critic (A2C)**

Each model will be trained in an identical environment with an action space of **Buy / Sell / Hold**, and compared against traditional TA-based strategies using standard financial performance metrics.

---

##  Objectives
- Develop RL-based trading agents that learn profitable trading behaviors from market data.  
- Compare RL strategies with rule-based TA methods (e.g., RSI, MACD, SMA crossovers).  
- Produce **day-by-day trading logs** to visualize RL agents’ decisions and corresponding profits.  
- Evaluate strategy performance using metrics such as **cumulative PnL** and **Sharpe ratio**.  

---

##  Data and Features

**Dataset:**  
- Historical **S&P 500** stock data (daily OHLCV) for the past **10 years**.  

**Technical Indicators (20 total):**  
RSI, MACD, Stochastic Oscillator, Bollinger Bands, SMA/EMA, ADX, CCI, OBV, ROC, ATR, and others.

**State Representations:**  
1. **Raw Price Input:** Close price, volume, and price history.  
2. **Indicator Input:** Vector of 20 normalized TA indicators.

---

##  Methodology

### 1. Data Preprocessing
- Collect and clean historical S&P 500 data.  
- Compute 20 TA indicators aligned with daily trading sessions.  
- Normalize and scale features for RL training.

### 2. Environment Design
- Define a simulated trading environment with:
  - **Action Space:** {Buy, Sell, Hold}
  - **Reward Function:** Daily **log return**
  - **Transaction Cost:** Applied to each trade for realism.  

### 3. Reinforcement Learning Models
- **DQN:** Value-based model using Q-learning with experience replay.  
- **PPO:** Policy-gradient algorithm optimizing a clipped objective function.  
- **A2C:** Actor–Critic method balancing policy and value updates.  
- Hyperparameter tuning for exploration (ε), learning rate, and discount factor (γ).

### 4. Traditional TA Baseline
- Implement classical TA-based trading rules such as:
  - RSI oversold/overbought crossover.  
  - MACD signal-line strategy.  
  - SMA20 vs SMA50 crossover trend following.  
- Evaluate these strategies on the same dataset and time period.

### 5. Evaluation Metrics
| Metric | Description |
|---------|--------------|
| **Cumulative PnL** | Total profit/loss over test period |
| **Sharpe Ratio** | Annualized risk-adjusted return |
| **Max Drawdown** | Largest equity decline from peak |
| **Win Rate** | Percentage of profitable trades |
| **Directional Accuracy** | Accuracy of predicting next price move |

### 6. Interpretability
- Generate **trading logs** showing for each day:
  - Date, Price, Action (Buy/Sell/Hold)
  - RL Agent Decision Probability
  - Cumulative Portfolio Value  
- Visualize **PnL curves** and **Sharpe ratio comparison charts**.

---

##  Expected Outcomes
- Quantitative comparison of **RL vs TA** trading performance.  
- Insight into how different RL architectures behave in dynamic financial markets.  
- Identification of conditions where RL methods outperform (or underperform) rule-based TA systems.  
- Evaluation of **explainability** through detailed trade logs.

---

##  Deliverables
- Source code implementing:
  - DQN, PPO, and Actor–Critic agents.
  - Backtesting environment for S&P 500 data.
  - TA-based baseline strategies.
- CSV and visual trading logs (Buy/Sell/Hold per day).  
- Evaluation dashboard with cumulative returns and Sharpe ratios.  
- Final report summarizing results, limitations, and potential extensions.

---


