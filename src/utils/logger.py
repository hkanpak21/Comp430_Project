import logging
import os
import csv
from datetime import datetime

def setup_logger(log_dir, log_filename, args):
    """Sets up logging to console and file."""
    log_filepath = os.path.join(log_dir, log_filename)
    file_exists = os.path.isfile(log_filepath)

    # Basic logger setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Log to console
        ]
    )

    # Add file handler for CSV logging
    logger = logging.getLogger(__name__)
    # Don't add file handler if logger already has handlers (e.g., during reload)
    if not logger.handlers or len(logger.handlers) <= 1: # Check if only StreamHandler exists
        file_handler = logging.FileHandler(log_filepath, mode='a') # Append mode
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # We won't add the file handler directly to the logger, as we'll handle CSV writing separately
        # logger.addHandler(file_handler)
        pass # We will manually write to CSV

    # Write header row if the file is new
    if not file_exists:
        header = ['timestamp'] + list(vars(args).keys()) + ['round', 'train_loss', 'accuracy', 'epsilon_per_round', 'total_epsilon']
        with open(log_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    return logger, log_filepath

def log_results(log_filepath, args, round_num, train_loss, accuracy, epsilon_per_round, total_epsilon):
    """Logs results for a specific round to the CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = [timestamp] + list(vars(args).values()) + [
        round_num, f'{train_loss:.4f}' if train_loss is not None else 'N/A',
        f'{accuracy:.4f}' if accuracy is not None else 'N/A',
        f'{epsilon_per_round:.4f}' if epsilon_per_round is not None else 'N/A',
        f'{total_epsilon:.4f}' if total_epsilon is not None else 'N/A'
    ]
    try:
        with open(log_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_entry)
    except IOError as e:
        logging.error(f"Error writing to log file {log_filepath}: {e}") 