"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    numeric_df = df.select_dtypes(include=np.number)

    summary = numeric_df.agg(['count', 'mean', 'median', 'std', 'min', 'max'])

    os.makedirs("output", exist_ok=True)
    summary.to_csv("output/summary.csv")

    return summary


def plot_distributions(df, columns, output_path):
    plt.figure(figsize=(10, 8))

    for i, col in enumerate(columns):
        plt.subplot(2, 2, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation(df, output_path):
    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr(method='pearson')

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    os.makedirs("output", exist_ok=True)

    # Load data
    df = pd.read_csv("data/sample_sales.csv")

    # Task 1
    compute_summary(df)

    # Task 2 — choose 4 numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns[:4]
    plot_distributions(df, numeric_cols, "output/distributions.png")

    # Task 3
    plot_correlation(df, "output/correlation.png")


if __name__ == "__main__":
    main()
