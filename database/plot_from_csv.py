import os
import pandas as pd
import matplotlib.pyplot as plt


# --- Paths to your results files ---
RESULT_FILES = {
    "DQN (Indicators)": "output/ml_outputs_dqn_indicators/full_simulation_AAPL.csv",
    "PPO (Indicators)": "output/ml_outputs_ppo_indicators/full_simulation_AAPL.csv",
    "DQN (Raw Price)": "output/ml_outputs_dqn_raw_vs_ta/full_simulation_AAPL_21d.csv",
    "PPO (Raw Price)": "output/ml_outputs_ppo_raw_vs_ta/full_simulation_AAPL_21d.csv",
}

# --- Column names from your simulation files ---
# These are the names of the portfolio value columns in your CSVs.
COLUMN_MAPPING = {
    "DQN (Indicators)": "portfolio_value_DQN_Paper",
    "PPO (Indicators)": "portfolio_value_PPO_Paper",
    "DQN (Raw Price)": "portfolio_value_DQN_RawPrice",
    "PPO (Raw Price)": "portfolio_value_PPO_RawPrice",
}


# Output file for the plot
OUTPUT_IMAGE_FILE = "performance_comparison.png"

# ==================================

def plot_performance_comparison(results_config, column_map, output_file):
    """
    Reads multiple simulation CSVs and plots their performance on a single chart.
    """
    print("[info] Starting performance plot generation...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    all_series = {}
    
    # Load and process each strategy's results
    for strategy_name, file_path in results_config.items():
        if not os.path.exists(file_path):
            print(f"[warn] File not found, skipping '{strategy_name}': {file_path}")
            continue
            
        print(f"[load] Reading results for '{strategy_name}' from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        col_name = column_map.get(strategy_name)
        if col_name not in df.columns:
            print(f"[warn] Column '{col_name}' not found in {file_path}. Skipping.")
            continue
            
        # Normalize the series to start at 1.0 to show growth of $1
        normalized_series = df[col_name] / df[col_name].iloc[0]
        all_series[strategy_name] = normalized_series

    # Load the Buy and Hold benchmark from the first available file
    bh_loaded = False
    for file_path in results_config.values():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            break
    
    if not all_series:
        print("[error] No data was loaded. Please check your file paths and column names.")
        return

    # Plot all loaded series
    for name, series in all_series.items():
        ax.plot(series.index, series.values, label=name, lw=2)

    ax.set_title("Performance Comparison of RL Trading Strategies", fontsize=18)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Growth (Normalized to $1)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Improve date formatting on the x-axis
    fig.autofmt_xdate()

    # Save the plot to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[save] Performance plot saved to: {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Before running, make sure you have matplotlib installed:
    # pip install matplotlib
    
    plot_performance_comparison(RESULT_FILES, COLUMN_MAPPING, OUTPUT_IMAGE_FILE)
