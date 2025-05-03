# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
log_file = 'experiments/results/experiment_log.csv'
output_dir = 'figures'
run_identifier_column = 'timestamp' # Assuming timestamp uniquely identifies a run
metrics_to_plot = {
    'accuracy': 'Test Accuracy (%)',
    'train_loss': 'Average Train Loss',
    'current_sigma': 'Noise Multiplier (Sigma)',
    'current_C': 'Clipping Norm (C)'
}
round_column = 'round'

# --- Create output directory ---
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
try:
    df = pd.read_csv(log_file)
    print(f"Successfully loaded data from {log_file}")
except FileNotFoundError:
    print(f"Error: Log file not found at {log_file}")
    exit(1)
except Exception as e:
    print(f"Error loading log file: {e}")
    exit(1)

# --- Identify the latest run ---
# Assuming the last entry corresponds to the end of the latest run
if run_identifier_column not in df.columns:
     print(f"Error: Run identifier column '{run_identifier_column}' not found in the CSV.")
     # Fallback: assume the last N rows are the latest run if 'epochs' is present
     if 'epochs' in df.columns:
         try:
             latest_epochs = int(df['epochs'].iloc[-1])
             print(f"Warning: Timestamp column not found. Assuming last {latest_epochs} rows are the latest run based on 'epochs' column.")
             latest_run_df = df.tail(latest_epochs)
         except (ValueError, KeyError, IndexError):
              print("Error: Could not determine latest run. Plotting all data.")
              latest_run_df = df # Plot everything if fallback fails
     else:
          print("Error: Cannot determine the latest run without timestamp or epochs column. Plotting all data.")
          latest_run_df = df # Plot everything
else:
    # Instead of filtering by the exact last timestamp,
    # determine the number of epochs in the last run and take the tail.
    try:
        latest_epochs = int(df['epochs'].iloc[-1])
        if latest_epochs > 0 and latest_epochs <= len(df):
             print(f"Identified latest run based on the last {latest_epochs} entries (epochs reported in the last row).")
             latest_run_df = df.tail(latest_epochs).copy() # Use tail to get the last N rows
        else:
             print(f"Warning: Invalid number of epochs ({latest_epochs}) found in the last row or log file too short. Plotting all data.")
             latest_run_df = df # Plot everything as a fallback
    except (ValueError, KeyError, IndexError):
         print("Error: Could not determine epochs from the last row. Plotting all data.")
         latest_run_df = df # Plot everything if epochs column is missing/invalid

# Ensure round column is suitable for plotting
if round_column not in latest_run_df.columns:
    print(f"Error: Round column '{round_column}' not found.")
    exit(1)
latest_run_df[round_column] = pd.to_numeric(latest_run_df[round_column], errors='coerce')
latest_run_df = latest_run_df.dropna(subset=[round_column])
latest_run_df = latest_run_df.sort_values(by=round_column)


# --- Generate Plots ---
print(f"Generating plots for {len(latest_run_df)} data points...")

for metric_col, plot_title in metrics_to_plot.items():
    if metric_col in latest_run_df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(latest_run_df[round_column], latest_run_df[metric_col], marker='o', linestyle='-')
        plt.title(f'{plot_title} vs. Communication Round')
        plt.xlabel('Communication Round (Epoch)')
        plt.ylabel(plot_title)
        plt.grid(True)
        plt.xticks(latest_run_df[round_column].unique()) # Ensure integer ticks for rounds

        # --- Save Plot ---
        output_filename = os.path.join(output_dir, f'{metric_col}_vs_round3.png')
        try:
            plt.savefig(output_filename)
            print(f"Saved plot: {output_filename}")
        except Exception as e:
            print(f"Error saving plot {output_filename}: {e}")
        plt.close() # Close the plot to free memory
    else:
        print(f"Warning: Metric column '{metric_col}' not found in the log file. Skipping plot.")

print("Plotting complete.") 