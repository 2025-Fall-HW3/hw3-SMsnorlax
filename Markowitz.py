"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust = False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""

class EqualWeightPortfolio:
    def __init__(self, exclude):
        # 要排除的資產（這邊會是 "SPY"）
        self.exclude = exclude

    def calculate_weights(self):
        # 取出除了 exclude 以外的資產
        assets = df.columns[df.columns != self.exclude]

        # 建一個全為 0 的 DataFrame，index 跟 df 一樣，columns 也跟 df 一樣
        self.portfolio_weights = pd.DataFrame(
            0.0, index=df.index, columns=df.columns
        )

        # 等權重：對每一天、對每個 sector（不含 SPY），給 1 / 資產數
        equal_weight = 1.0 / len(assets)
        self.portfolio_weights.loc[:, assets] = equal_weight

        # 補齊可能的 NaN（其實這裡大多不會有，但照原本框架做）
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # 確保權重已經算好
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # 計算報酬
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # 確保報酬已經算好
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""
class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # 除了 SPY 以外的資產
        assets = df.columns[df.columns != self.exclude]

        # 權重表：一開始都設 0
        self.portfolio_weights = pd.DataFrame(
            0.0, index=df.index, columns=df.columns
        )

        # 從 lookback 之後的日期開始算 rolling volatility
        for i in range(self.lookback + 1, len(df)):
            # 取過去 lookback 天的報酬
            window_ret = df_returns[assets].iloc[i - self.lookback : i]

            # 每個資產的波動度（標準差）
            sigma = window_ret.std().values

            # 避免除以 0：如果有 0 就換成一個很小的數
            sigma = np.where(sigma == 0, 1e-6, sigma)

            # inverse volatility
            inv_sigma = 1.0 / sigma
            weights = inv_sigma / inv_sigma.sum()

            # 存到對應日期那一列
            self.portfolio_weights.loc[df.index[i], assets] = weights

        # 把一開始的 NaN 往前填 & 補 0
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns

"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)
    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values   # 協方差矩陣 Σ
        mu = R_n.mean().values     # 期望報酬 μ
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                # === Decision variable: w (長度 n 的權重向量) ===
                # long-only: w >= 0，且已經設 ub=1
                w = model.addMVar(n, name="w", lb=0.0, ub=1.0)

                # === 目標函數: maximize w^T mu - gamma/2 * w^T Sigma w ===
                # 線性部分
                lin_term = mu @ w
                # 二次部分
                quad_term = w @ Sigma @ w
                model.setObjective(lin_term - gamma / 2.0 * quad_term,
                                   gp.GRB.MAXIMIZE)

                # === 約束條件: sum_i w_i = 1 (no leverage) ===
                model.addConstr(w.sum() == 1.0, name="budget")

                model.optimize()

                # 取解
                solution = np.zeros(n)
                if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    solution = w.X  # MVar 直接拿解

        return solution

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)
