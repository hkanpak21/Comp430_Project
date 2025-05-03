import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the path to your log file
log_filepath = 'experiments/results/experiment_log.csv'

# Define the expected column names explicitly
column_names = [
    "timestamp","seed","device","log_dir","log_filename","dataset","data_root",
    "data_distribution","non_iid_alpha","num_clients","clients_per_round",
    "model","split_layer","epochs","local_epochs","optimizer","lr","batch_size",
    "dp_mode","target_epsilon","target_delta","noise_multiplier","max_grad_norm",
    "adaptive_clipping_quantile","adaptive_noise_decay","awdp_layerwise",
    "trusted_client_id","feedback_metric","adaptive_step_size","min_sigma",
    "max_sigma","min_C","max_C","round","train_loss","accuracy",
    "epsilon_per_round","total_epsilon","current_sigma","current_C","avg_grad_norm"
]

try:
    # Read the CSV file, specifying column names and NO header (header=None)
    # Also skip bad lines encountered during parsing
    df = pd.read_csv(log_filepath, names=column_names, header=None, on_bad_lines='skip')

    # --- Identify the specific experiment run (rows 71-90, indices 70-89) ---
    start_index = 70 # Row 71 is index 70
    end_index = 90   # iloc end index is exclusive, so use 90 to include index 89 (Row 90)
    if start_index >= len(df) or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid index range ({start_index+1}-{end_index}) for the dataframe with {len(df)} rows.")
        exit()
        
    specific_run_df = df.iloc[start_index:end_index].copy()
    print(f"Plotting data from rows {start_index+1} to {end_index} (Timestamp: {specific_run_df.iloc[0]['timestamp']})" if not specific_run_df.empty else "Selected rows are empty.")

    if specific_run_df.empty:
        exit()
        
    print(f"Configuration for this run (example row):\n{specific_run_df.iloc[0]}\n")

    # --- Data Cleaning ---
    # Convert 'avg_grad_norm' to numeric, handling potential 'N/A' or errors
    specific_run_df['avg_grad_norm'] = pd.to_numeric(specific_run_df['avg_grad_norm'], errors='coerce')
    # Convert 'round' to numeric (should be clean, but good practice)
    specific_run_df['round'] = pd.to_numeric(specific_run_df['round'], errors='coerce')

    # Drop rows where avg_grad_norm could not be converted (became NaN)
    specific_run_df.dropna(subset=['avg_grad_norm', 'round'], inplace=True)

    # --- Plotting ---
    if not specific_run_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(specific_run_df['round'], specific_run_df['avg_grad_norm'], marker='o', linestyle='-')
        plt.title(f'Average Client Gradient Norm vs. Round (Rows {start_index+1}-{end_index})')
        plt.xlabel('Round')
        plt.ylabel('Average Gradient Norm')
        plt.grid(True)
        # Ensure ticks are integers and reasonably spaced
        min_round = int(specific_run_df['round'].min())
        max_round = int(specific_run_df['round'].max())
        step = max(1, int((max_round - min_round) / 10)) # Aim for ~10 ticks
        plt.xticks(np.arange(min_round, max_round + 1, step=step))

        plot_filename = f'run_rows_{start_index+1}_{end_index}_gradient_norm_plot3.png'
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        # plt.show() # Calling plt.show() might not work in this environment, saving instead.
    else:
        print("No valid data found for the specified rows to plot gradient norms.")

except FileNotFoundError:
    print(f"Error: Log file not found at {log_filepath}")
except Exception as e:
    print(f"An error occurred: {e}") 