import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_q_matrix(Q_df, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(Q_df, annot=True, cmap="Blues", fmt=".2f", linewidths=.5)
    plt.title("Q Matrix: User Journey Flow Probabilities")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_r_matrix(R_df, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(R_df, annot=True, cmap="Greens", fmt=".2f", linewidths=.5)
    plt.title("R Matrix: Conversion vs. Drop-off Probabilities")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_removal_effects(removal_series, output_path):
    df = removal_series.sort_values(ascending=True).reset_index()
    df.columns = ['Channel', 'Removal Effect']
    
    plt.figure(figsize=(10, 6))
    # FIX: Assigned y to hue and set legend=False
    sns.barplot(x='Removal Effect', y='Channel', data=df, hue='Channel', palette="viridis", legend=False)
    plt.title("Channel Value (Removal Effect on Conversions)")
    plt.xlabel("Probability Decrease if Channel Removed")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_conversion_probs(conv_prob_series, output_path):
    df = conv_prob_series.sort_values(ascending=True).reset_index()
    df.columns = ['Channel', 'Conversion Probability']
    
    plt.figure(figsize=(10, 6))
    # FIX: Assigned y to hue and set legend=False
    sns.barplot(x='Conversion Probability', y='Channel', data=df, hue='Channel', palette="magma", legend=False)
    plt.title("Direct Conversion Probability by Channel")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()