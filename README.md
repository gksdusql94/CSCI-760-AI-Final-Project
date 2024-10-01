# 📈 Optimizing Stock Trading Strategies with Various Algorithms

## 🌐 Overview

This project explores and compares the effectiveness of several AI algorithms in stock trading, focusing on maximizing financial returns or minimizing losses. Algorithms such as **XGBoost**, **Monte Carlo simulations**, **Q-learning**, and **Deep Reinforcement Learning** will be implemented to develop robust trading strategies across various market conditions, including stocks and cryptocurrencies.

## 👥  Team Roles

- XGBoost implementation, dataset preparation, and evaluation.(Yeonbi)
- Q-learning for value-based trading strategies.
- Deep Reinforcement Learning using policy gradients, focusing on fully connected layers and CNNs.
- **Shared Tasks**: Feature engineering, data preprocessing, and evaluation.

## 📝 Project Description

We aim to optimize decision-making in stock trading using AI techniques like reinforcement learning. By applying various algorithms, we will analyze their strengths and weaknesses in trading decisions across diverse market conditions (e.g., Bitcoin, Apple). Our evaluation will focus on metrics such as total profit/loss and risk-adjusted returns.

## 📦  Dataset Preparation (My Contribution)

I prepared the stock price dataset, which includes the columns: Open, High, Low, Close, and Volume. We created a target variable, Target, to indicate whether the Close price will rise. The data was used to generate input and target data for the GRU model.
- **Data**: [BTC_5min.txt, AAPL_5min.txt](https://drive.google.com/drive/folders/1htN-2fW1qNGrNnSYx5oeNW2NNZr4Ntle?usp=sharing)

```python
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)  # Drop last row to handle NaN values

window_size = 10  # Window size for the time series data
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X, y = [], []

for i in range(window_size, len(df)):
    X.append(df[features].iloc[i-window_size:i].values)
    y.append(df['Target'].iloc[i])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 📊 Results & Discussion

### # Q-Learning
In stock trading, the environment changes rapidly and is influenced by various factors, which makes it difficult for traders to adopt a single strategy. Reinforcement learning, particularly **Q-learning**, provides a framework for creating adaptive trading algorithms based on real-time variables such as closing prices and stock ownership.

**Q-learning** is a model-free reinforcement learning technique used to make sequential decisions by mapping states to actions. It optimizes cumulative rewards over time. In this project, we employed Q-learning to create an intelligent trading agent that learns buy and sell decisions based on historical data.

### # Q-Learning Experiment Setup
- **Dataset**: Historical stock data of **Apple (AAPL)** from 2005/01/03 to 2022/09/26 (~700,000 records).
- We created two smaller datasets from the original: **AAPL_300k** (300,000 rows) and **AAPL_per_day** (aggregated daily data).
- **Initial Parameters**:
  - Alpha: 0.1, Gamma: 0.95, Epsilon: 0.1, Epochs: 10
  - Initial Capital: $1,000

### # Final Capital Results from Q-Learning
| Dataset      | Final Capital ($) |
|--------------|-------------------|
| **AAPL**     | 8,102.95          |
| **AAPL_300k**| 265,946.25        |
| **AAPL_per_day** | 157,894.73     |

### # Agent’s Actions Over AAPL Close Price
We plotted Q-values over the close price for each dataset and selected actions using the `argmax` function to observe the agent’s behavior.

**Figures**:  
- AAPL_300k  
- AAPL  
- AAPL_per_day  

---

### # SDE Analysis
The focus was on analyzing stock price movements using **Stochastic Differential Equations (SDEs)** and testing **Proximal Policy Optimization (PPO)** algorithms to forecast trends. Multiple configurations of **mean (mu)** and **standard deviation (SD)** for volatility modeling were tested:

- **SDE Projection 1**: Mu = 0.001, SD = 0.01
- **SDE Projection 2**: Mu = 0.0, SD = 0.1
- **SDE Projection 3**: Mu = 0.001, SD = 0.1

Additionally, we identified a recurring pattern where the market trends upward or downward for three consecutive days and adjusted prices accordingly. The **PPO algorithm**, although positive in cumulative P&L, struggled with oscillating behavior, often converging into suboptimal actions (buy/sell).

---

### # Results & Discussion
- The first model (**YB XGBoost 1**) achieved a **61.11% accuracy** on the test data, outperforming the standalone GRU model.
- The second model (**YB XGBoost 2**), despite using more epochs and layers, resulted in a lower **55% accuracy**, likely due to overfitting or increased complexity. Future improvements could include optimizing hyperparameters and testing different architectures.
- The results show that Q-learning benefits significantly from more extensive datasets, as seen from the final capital results. Q-learning produced the best results when applied to the complete dataset, especially with Apple stock, which showed a strong upward trend.
- The **PPO model** displayed sensitivity to **mu** and **SD**. When SD was large, the model struggled to learn effectively, often oscillating and locking into suboptimal actions. The market’s behavior over three days influenced the model's ability to anticipate changes, but balancing actions (buy/sell) remained challenging.
