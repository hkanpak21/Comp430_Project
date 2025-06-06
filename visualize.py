import os
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
import matplotlib.gridspec as gridspec
from itertools import combinations

# -------------------------------------------------------------------------
# 0.  File location – adapt if you store the CSV elsewhere
# -------------------------------------------------------------------------
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Try to find the CSV file in the experiments directory
fname = os.path.join(script_dir, 'experiments', 'Colab Deney Sonuçları.csv')

# If not found, look for other possible locations
if not os.path.exists(fname):
    # Check current working directory
    fname = 'Colab Deney Sonuçları.csv'
    if not os.path.exists(fname):
        # Try other potential locations
        potential_paths = [
            os.path.join(script_dir, 'experiments', 'Colab Deney Sonuçları.csv'),
            os.path.join(os.getcwd(), 'experiments', 'Colab Deney Sonuçları.csv'),
            '/experiments/Colab Deney Sonuçları.csv'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                fname = path
                break

# Print the file path we're using
print(f"Using CSV file: {fname}")
print(f"File exists: {os.path.exists(fname)}")

# Read the experiment table ------------------------------------------------
df = pd.read_csv(fname)

# Drop the success column as it's irrelevant
if 'success' in df.columns:
    df = df.drop('success', axis=1)
    
# Define parameters of interest for analysis
parameters = ['model', 'cut_layer', 'lr', 'batch_size', 'num_clients', 'num_rounds', 'noise_multiplier', 'initial_sigma']
metrics = ['final_acc', 'elapsed_time']

# Print dataset overview
print(f"\nDataset Overview:")
print(f"Total experiments: {len(df)}")
print("\nUnique values per parameter:")
for param in parameters:
    unique_vals = df[param].unique()
    print(f"{param}: {len(unique_vals)} unique values - {sorted(unique_vals)}")

# Function to find experiment groups where only one parameter varies
def find_controlled_experiments(df, target_param, control_params=None):
    """Find groups of experiments where only target_param varies and others are constant"""
    if control_params is None:
        control_params = [p for p in parameters if p != target_param]
    
    # Group by all control parameters
    grouped = df.groupby(control_params)
    
    # Find groups with multiple experiments (meaning target_param varies)
    valid_groups = []
    for name, group in grouped:
        if len(group) > 1 and group[target_param].nunique() > 1:
            valid_groups.append(group)
    
    return valid_groups

# -------------------------------------------------------------------------
# 1.  Meta-style figure defaults (≈ what Meta / FAIR use in papers)
# -------------------------------------------------------------------------
meta_rc = {
    # paper-sized figure
    'figure.figsize'   : (10, 8),   # inches - larger for multi-plot figures
    'figure.dpi'       : 300,
    # font
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 10,
    'legend.fontsize'  : 9,
    # lines & markers
    'lines.linewidth'  : 1.5,
    'lines.markersize' : 8,        # Slightly larger markers for better visibility
    # axes aesthetics
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,      # Enable grid for better readability
    'grid.alpha'       : 0.3,       # Subtle grid
    'xtick.direction'  : 'out',
    'ytick.direction'  : 'out',
    # use a light (paper-friendly) background
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
}
plt.rcParams.update(meta_rc)

# Create a directory for figures if it doesn't exist
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Create subdirectories for different types of visualizations
localized_dir = os.path.join(figures_dir, 'localized')
merged_dir = os.path.join(figures_dir, 'merged')
os.makedirs(localized_dir, exist_ok=True)
os.makedirs(merged_dir, exist_ok=True)

# Color palettes for different models and groups
colors_dict = {
    'SimpleDNN': '#e41a1c',  # red
    'SimpleCNN': '#377eb8',  # blue
    'default': '#4daf4a'      # green (for any other model)
}

# Get unique models for consistent coloring
unique_models = df['model'].unique()
model_colors = {}
for model in unique_models:
    if model in colors_dict:
        model_colors[model] = colors_dict[model]
    else:
        model_colors[model] = colors_dict['default']
        
# Color palette for different experiment groups
group_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
]
        
# Function to create a localized comparison figure
def create_localized_figure(group_df, target_param, metric, group_id, fixed_params=None):
    """Create a figure showing the relationship between target_param and metric
    for a group of experiments where only target_param varies."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort the dataframe by the target parameter for better visualization
    group_df = group_df.sort_values(by=target_param)
    
    # Plot the data points
    ax.plot(group_df[target_param], group_df[metric], 'o-', 
            markersize=8, linewidth=2, color='#377eb8')
    
    # Add data labels
    for i, row in group_df.iterrows():
        ax.text(row[target_param], row[metric], f'{row[metric]:.2f}', 
                fontsize=9, ha='center', va='bottom')
    
    # Set labels and title
    ax.set_xlabel(target_param.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    
    # Create title with fixed parameters information
    title = f'{metric.replace("_", " ").title()} vs {target_param.replace("_", " ").title()}'
    ax.set_title(title)
    
    # Add fixed parameters as text annotation
    if fixed_params:
        fixed_text = "\n".join([f"{p.replace('_', ' ').title()}: {group_df[p].iloc[0]}" 
                             for p in fixed_params])
        ax.text(0.02, 0.02, fixed_text, transform=ax.transAxes, 
                fontsize=8, va='bottom', ha='left', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    # Save the figure
    filename = f"{target_param}_vs_{metric}_group{group_id}.png"
    fig_path = os.path.join(localized_dir, filename)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path

# Function to create a multi-metric comparison figure
def create_multi_metric_figure(group_df, target_param, group_id, fixed_params=None):
    """Create a figure showing the relationship between target_param and multiple metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Sort the dataframe by the target parameter
    group_df = group_df.sort_values(by=target_param)
    
    # Plot accuracy on the first axis
    ax1.plot(group_df[target_param], group_df['final_acc'], 'o-', 
             markersize=8, linewidth=2, color='#377eb8')
    
    # Add data labels
    for i, row in group_df.iterrows():
        ax1.text(row[target_param], row['final_acc'], f'{row["final_acc"]:.2f}', 
                 fontsize=9, ha='center', va='bottom')
    
    ax1.set_xlabel(target_param.replace('_', ' ').title())
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_title(f'Accuracy vs {target_param.replace("_", " ").title()}')
    
    # Plot training time on the second axis
    ax2.plot(group_df[target_param], group_df['elapsed_time'], 'o-', 
             markersize=8, linewidth=2, color='#e41a1c')
    
    # Add data labels
    for i, row in group_df.iterrows():
        ax2.text(row[target_param], row['elapsed_time'], f'{row["elapsed_time"]:.0f}s', 
                 fontsize=9, ha='center', va='bottom')
    
    ax2.set_xlabel(target_param.replace('_', ' ').title())
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title(f'Training Time vs {target_param.replace("_", " ").title()}')
    
    # Add fixed parameters as text annotation
    if fixed_params:
        fixed_text = "\n".join([f"{p.replace('_', ' ').title()}: {group_df[p].iloc[0]}" 
                             for p in fixed_params])
        fig.text(0.01, 0.02, fixed_text, fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    # Set the figure title
    fig.suptitle(f'Impact of {target_param.replace("_", " ").title()} (Group {group_id})', fontsize=14)
    
    # Save the figure
    filename = f"{target_param}_multi_metric_group{group_id}.png"
    fig_path = os.path.join(localized_dir, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path

# Function to create a dual-parameter comparison figure
def create_dual_param_figure(df, param1, param2, metric):
    """Create a figure showing how two parameters jointly affect a metric"""
    
    # Group by the two parameters
    pivot_table = df.pivot_table(index=param1, columns=param2, values=metric, aggfunc='mean')
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a heatmap
    im = ax.imshow(pivot_table, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    
    # Rotate the x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom")
    
    # Add text annotations in each cell
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            try:
                value = pivot_table.iloc[i, j]
                if not pd.isna(value):
                    text = ax.text(j, i, f"{value:.2f}",
                                ha="center", va="center", color="white" if value > pivot_table.mean().mean() else "black")
            except:
                pass
    
    # Set labels and title
    ax.set_xlabel(param2.replace('_', ' ').title())
    ax.set_ylabel(param1.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} by {param1.replace("_", " ").title()} and {param2.replace("_", " ").title()}')
    
    # Save the figure
    filename = f"{param1}_and_{param2}_vs_{metric}.png"
    fig_path = os.path.join(localized_dir, filename)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path

# Function to create a merged comparison figure with multiple experiment groups
def create_merged_figure(groups, target_param, metric, fixed_params_list=None, group_labels=None):
    """Create a figure showing multiple experiment groups with different colors"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a legend handler
    legend_elements = []
    
    # Plot each group with a different color
    for i, group_df in enumerate(groups):
        # Sort the dataframe by the target parameter
        group_df = group_df.sort_values(by=target_param)
        
        # Get color for this group
        color = group_colors[i % len(group_colors)]
        
        # Create label for this group
        if group_labels and i < len(group_labels):
            label = group_labels[i]
        elif fixed_params_list and i < len(fixed_params_list):
            # Create a label from the fixed parameters
            fixed_params = fixed_params_list[i]
            label_parts = []
            for p in fixed_params:
                val = group_df[p].iloc[0]
                # Format the value (shorten if it's a float)
                if isinstance(val, float) and val != int(val):
                    val_str = f"{val:.3f}".rstrip('0').rstrip('.')
                else:
                    val_str = str(val)
                label_parts.append(f"{p.replace('_', ' ')}={val_str}")
            label = ", ".join(label_parts)
        else:
            label = f"Group {i+1}"
        
        # Plot the data points
        line = ax.plot(group_df[target_param], group_df[metric], 'o-', 
                markersize=8, linewidth=2, color=color, label=label)
        
        # Add data labels if there aren't too many points
        if len(group_df) <= 8:
            for _, row in group_df.iterrows():
                ax.text(row[target_param], row[metric], f'{row[metric]:.2f}', 
                        fontsize=8, ha='center', va='bottom', color=color)
    
    # Set labels and title
    ax.set_xlabel(target_param.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} vs {target_param.replace("_", " ").title()} (Multiple Groups)')
    
    # Add legend
    ax.legend(loc='best', frameon=True, fontsize=9)
    
    # Save the figure
    filename = f"merged_{target_param}_vs_{metric}.png"
    fig_path = os.path.join(merged_dir, filename)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path

# Function to create a merged multi-metric comparison figure
def create_merged_multi_metric_figure(groups, target_param, fixed_params_list=None, group_labels=None):
    """Create a figure showing multiple experiment groups with different colors for multiple metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot each group with a different color
    for i, group_df in enumerate(groups):
        # Sort the dataframe by the target parameter
        group_df = group_df.sort_values(by=target_param)
        
        # Get color for this group
        color = group_colors[i % len(group_colors)]
        
        # Create label for this group
        if group_labels and i < len(group_labels):
            label = group_labels[i]
        elif fixed_params_list and i < len(fixed_params_list):
            # Create a label from the fixed parameters
            fixed_params = fixed_params_list[i]
            label_parts = []
            for p in fixed_params:
                val = group_df[p].iloc[0]
                # Format the value (shorten if it's a float)
                if isinstance(val, float) and val != int(val):
                    val_str = f"{val:.3f}".rstrip('0').rstrip('.')
                else:
                    val_str = str(val)
                label_parts.append(f"{p.replace('_', ' ')}={val_str}")
            label = ", ".join(label_parts)
        else:
            label = f"Group {i+1}"
        
        # Plot accuracy on the first axis
        ax1.plot(group_df[target_param], group_df['final_acc'], 'o-', 
                 markersize=8, linewidth=2, color=color, label=label)
        
        # Plot training time on the second axis
        ax2.plot(group_df[target_param], group_df['elapsed_time'], 'o-', 
                 markersize=8, linewidth=2, color=color, label=label)
    
    # Set labels and titles
    ax1.set_xlabel(target_param.replace('_', ' ').title())
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_title(f'Accuracy vs {target_param.replace("_", " ").title()}')
    ax1.legend(loc='best', frameon=True, fontsize=9)
    
    ax2.set_xlabel(target_param.replace('_', ' ').title())
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title(f'Training Time vs {target_param.replace("_", " ").title()}')
    ax2.legend(loc='best', frameon=True, fontsize=9)
    
    # Set the figure title
    fig.suptitle(f'Impact of {target_param.replace("_", " ").title()} Across Multiple Experiment Groups', fontsize=14)
    
    # Save the figure
    filename = f"merged_{target_param}_multi_metric.png"
    fig_path = os.path.join(merged_dir, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path

# Function to create and save a figure
def create_figure(x_col, y_col, color_col=None, title=None, filename=None, hue_col=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if hue_col is not None:
        # Plot with categorical coloring based on hue_col
        for val, group in df.groupby(hue_col):
            ax.scatter(
                group[x_col], 
                group[y_col],
                label=f'{hue_col}={val}',
                color=model_colors.get(val, '#333333') if hue_col == 'model' else None,
                edgecolor='black',
                alpha=0.7
            )
        ax.legend(frameon=True, title=hue_col)
    elif color_col is not None:
        # Plot with continuous coloring based on color_col
        norm = colors.Normalize(df[color_col].min(), df[color_col].max())
        sc = ax.scatter(
            df[x_col], 
            df[y_col],
            c=df[color_col],
            cmap='viridis',
            edgecolor='black',
            alpha=0.7
        )
        plt.colorbar(sc, ax=ax, label=color_col)
    else:
        # Simple scatter plot
        ax.scatter(
            df[x_col], 
            df[y_col],
            edgecolor='black',
            alpha=0.7
        )
    
    # Add trendline
    try:
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        ax.plot(sorted(df[x_col]), p(sorted(df[x_col])), "--", color='#999999')
    except:
        pass  # Skip trendline if it can't be calculated
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}')
    
    fig.tight_layout()
    
    if filename:
        fig_path = os.path.join(figures_dir, filename)
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Saved figure: {fig_path}")
    else:
        fig_path = os.path.join(figures_dir, f"{y_col}_vs_{x_col}.png")
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Saved figure: {fig_path}")
    
    plt.close(fig)

# Function to create a multi-panel figure with 4 plots
def create_multipanel_figure(feature, filename=None):
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
    
    # Plot 1: Feature vs Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    for model, group in df.groupby('model'):
        ax1.scatter(
            group[feature], 
            group['final_acc'],
            label=model,
            color=model_colors.get(model, '#333333'),
            edgecolor='black',
            alpha=0.7
        )
    ax1.set_xlabel(feature.replace('_', ' ').title())
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_title(f'Accuracy vs {feature.replace("_", " ").title()}')
    ax1.legend(frameon=True, title='Model')
    
    # Plot 2: Feature vs Training Time
    ax2 = fig.add_subplot(gs[0, 1])
    for model, group in df.groupby('model'):
        ax2.scatter(
            group[feature], 
            group['elapsed_time'],
            label=model,
            color=model_colors.get(model, '#333333'),
            edgecolor='black',
            alpha=0.7
        )
    ax2.set_xlabel(feature.replace('_', ' ').title())
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title(f'Training Time vs {feature.replace("_", " ").title()}')
    ax2.legend(frameon=True, title='Model')
    
    # Plot 3: Feature vs Batch Size
    ax3 = fig.add_subplot(gs[1, 0])
    for model, group in df.groupby('model'):
        ax3.scatter(
            group[feature], 
            group['batch_size'],
            label=model,
            color=model_colors.get(model, '#333333'),
            edgecolor='black',
            alpha=0.7
        )
    ax3.set_xlabel(feature.replace('_', ' ').title())
    ax3.set_ylabel('Batch Size')
    ax3.set_title(f'Batch Size vs {feature.replace("_", " ").title()}')
    ax3.legend(frameon=True, title='Model')
    
    # Plot 4: Feature vs Learning Rate
    ax4 = fig.add_subplot(gs[1, 1])
    for model, group in df.groupby('model'):
        ax4.scatter(
            group[feature], 
            group['lr'],
            label=model,
            color=model_colors.get(model, '#333333'),
            edgecolor='black',
            alpha=0.7
        )
    ax4.set_xlabel(feature.replace('_', ' ').title())
    ax4.set_ylabel('Learning Rate')
    ax4.set_title(f'Learning Rate vs {feature.replace("_", " ").title()}')
    ax4.legend(frameon=True, title='Model')
    
    fig.suptitle(f'Impact of {feature.replace("_", " ").title()} on Training Metrics', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    if filename:
        fig_path = os.path.join(figures_dir, filename)
    else:
        fig_path = os.path.join(figures_dir, f"{feature}_multipanel.png")
    
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"Saved multipanel figure: {fig_path}")
    plt.close(fig)

# -------------------------------------------------------------------------
# 2. Create localized comparisons for specific parameters
# -------------------------------------------------------------------------

# Dictionary to store created figures by parameter
created_figures = {param: [] for param in parameters}
total_figures = 0

# -------------------------------------------------------------------------
# 2.1 Localized comparisons for Cut Layer
# -------------------------------------------------------------------------
print("\nCreating localized comparisons for Cut Layer...")
cut_layer_groups = find_controlled_experiments(df, 'cut_layer')
print(f"Found {len(cut_layer_groups)} experiment groups where only cut_layer varies")

for i, group in enumerate(cut_layer_groups):
    # Get the fixed parameters for this group
    fixed_params = [p for p in parameters if p != 'cut_layer' and group[p].nunique() == 1]
    
    # Create a multi-metric figure for this group
    fig_path = create_multi_metric_figure(group, 'cut_layer', i+1, fixed_params)
    created_figures['cut_layer'].append(fig_path)
    total_figures += 1
    
    # Also create individual metric figures
    for metric in metrics:
        fig_path = create_localized_figure(group, 'cut_layer', metric, i+1, fixed_params)
        created_figures['cut_layer'].append(fig_path)
        total_figures += 1

# -------------------------------------------------------------------------
# 2.2 Localized comparisons for Number of Clients
# -------------------------------------------------------------------------
print("\nCreating localized comparisons for Number of Clients...")
clients_groups = find_controlled_experiments(df, 'num_clients')
print(f"Found {len(clients_groups)} experiment groups where only num_clients varies")

for i, group in enumerate(clients_groups):
    # Get the fixed parameters for this group
    fixed_params = [p for p in parameters if p != 'num_clients' and group[p].nunique() == 1]
    
    # Create a multi-metric figure for this group
    fig_path = create_multi_metric_figure(group, 'num_clients', i+1, fixed_params)
    created_figures['num_clients'].append(fig_path)
    total_figures += 1
    
    # Also create individual metric figures
    for metric in metrics:
        fig_path = create_localized_figure(group, 'num_clients', metric, i+1, fixed_params)
        created_figures['num_clients'].append(fig_path)
        total_figures += 1

# -------------------------------------------------------------------------
# 2.3 Localized comparisons for Noise Multiplier
# -------------------------------------------------------------------------
print("\nCreating localized comparisons for Noise Multiplier...")
noise_groups = find_controlled_experiments(df, 'noise_multiplier')
print(f"Found {len(noise_groups)} experiment groups where only noise_multiplier varies")

for i, group in enumerate(noise_groups):
    # Get the fixed parameters for this group
    fixed_params = [p for p in parameters if p != 'noise_multiplier' and group[p].nunique() == 1]
    
    # Create a multi-metric figure for this group
    fig_path = create_multi_metric_figure(group, 'noise_multiplier', i+1, fixed_params)
    created_figures['noise_multiplier'].append(fig_path)
    total_figures += 1
    
    # Also create individual metric figures
    for metric in metrics:
        fig_path = create_localized_figure(group, 'noise_multiplier', metric, i+1, fixed_params)
        created_figures['noise_multiplier'].append(fig_path)
        total_figures += 1

# -------------------------------------------------------------------------
# 2.4 Localized comparisons for Batch Size
# -------------------------------------------------------------------------
print("\nCreating localized comparisons for Batch Size...")
batch_groups = find_controlled_experiments(df, 'batch_size')
print(f"Found {len(batch_groups)} experiment groups where only batch_size varies")

for i, group in enumerate(batch_groups):
    # Get the fixed parameters for this group
    fixed_params = [p for p in parameters if p != 'batch_size' and group[p].nunique() == 1]
    
    # Create a multi-metric figure for this group
    fig_path = create_multi_metric_figure(group, 'batch_size', i+1, fixed_params)
    created_figures['batch_size'].append(fig_path)
    total_figures += 1
    
    # Also create individual metric figures
    for metric in metrics:
        fig_path = create_localized_figure(group, 'batch_size', metric, i+1, fixed_params)
        created_figures['batch_size'].append(fig_path)
        total_figures += 1

# -------------------------------------------------------------------------
# 2.5 Localized comparisons for Learning Rate
# -------------------------------------------------------------------------
print("\nCreating localized comparisons for Learning Rate...")
lr_groups = find_controlled_experiments(df, 'lr')
print(f"Found {len(lr_groups)} experiment groups where only lr varies")

for i, group in enumerate(lr_groups):
    # Get the fixed parameters for this group
    fixed_params = [p for p in parameters if p != 'lr' and group[p].nunique() == 1]
    
    # Create a multi-metric figure for this group
    fig_path = create_multi_metric_figure(group, 'lr', i+1, fixed_params)
    created_figures['lr'].append(fig_path)
    total_figures += 1
    
    # Also create individual metric figures
    for metric in metrics:
        fig_path = create_localized_figure(group, 'lr', metric, i+1, fixed_params)
        created_figures['lr'].append(fig_path)
        total_figures += 1

# -------------------------------------------------------------------------
# 3. Create dual-parameter comparisons for key parameter pairs
# -------------------------------------------------------------------------
print("\nCreating dual-parameter comparisons...")

# Define key parameter pairs to analyze
param_pairs = [
    ('cut_layer', 'num_clients'),
    ('cut_layer', 'noise_multiplier'),
    ('num_clients', 'noise_multiplier'),
    ('batch_size', 'lr')
]

# Create dual-parameter visualizations
for param1, param2 in param_pairs:
    # Check if we have enough data for this pair
    if df.groupby([param1, param2]).size().reset_index().shape[0] > 3:
        for metric in metrics:
            try:
                fig_path = create_dual_param_figure(df, param1, param2, metric)
                print(f"Created dual-parameter figure: {fig_path}")
                total_figures += 1
            except Exception as e:
                print(f"Error creating dual-parameter figure for {param1} and {param2}: {e}")

# -------------------------------------------------------------------------
# 4. Create merged comparison figures
# -------------------------------------------------------------------------
print("\nCreating merged comparison figures...")

# Dictionary to store merged figures
merged_figures = {param: [] for param in parameters}

# -------------------------------------------------------------------------
# 4.1 Merged comparisons for Cut Layer
# -------------------------------------------------------------------------
if len(cut_layer_groups) > 1:
    print("\nCreating merged figures for Cut Layer...")
    
    # Get fixed parameters for each group
    fixed_params_list = []
    for group in cut_layer_groups:
        fixed_params = [p for p in parameters if p != 'cut_layer' and group[p].nunique() == 1]
        fixed_params_list.append(fixed_params)
    
    # Create merged figures
    for metric in metrics:
        fig_path = create_merged_figure(cut_layer_groups, 'cut_layer', metric, fixed_params_list)
        merged_figures['cut_layer'].append(fig_path)
        print(f"Created merged figure: {fig_path}")
    
    # Create multi-metric merged figure
    fig_path = create_merged_multi_metric_figure(cut_layer_groups, 'cut_layer', fixed_params_list)
    merged_figures['cut_layer'].append(fig_path)
    print(f"Created merged multi-metric figure: {fig_path}")

# -------------------------------------------------------------------------
# 4.2 Merged comparisons for Number of Clients
# -------------------------------------------------------------------------
if len(clients_groups) > 1:
    print("\nCreating merged figures for Number of Clients...")
    
    # Get fixed parameters for each group
    fixed_params_list = []
    for group in clients_groups:
        fixed_params = [p for p in parameters if p != 'num_clients' and group[p].nunique() == 1]
        fixed_params_list.append(fixed_params)
    
    # Create merged figures
    for metric in metrics:
        fig_path = create_merged_figure(clients_groups, 'num_clients', metric, fixed_params_list)
        merged_figures['num_clients'].append(fig_path)
        print(f"Created merged figure: {fig_path}")
    
    # Create multi-metric merged figure
    fig_path = create_merged_multi_metric_figure(clients_groups, 'num_clients', fixed_params_list)
    merged_figures['num_clients'].append(fig_path)
    print(f"Created merged multi-metric figure: {fig_path}")

# -------------------------------------------------------------------------
# 4.3 Merged comparisons for Batch Size
# -------------------------------------------------------------------------
if len(batch_groups) > 1:
    print("\nCreating merged figures for Batch Size...")
    
    # Get fixed parameters for each group
    fixed_params_list = []
    for group in batch_groups:
        fixed_params = [p for p in parameters if p != 'batch_size' and group[p].nunique() == 1]
        fixed_params_list.append(fixed_params)
    
    # Create merged figures
    for metric in metrics:
        fig_path = create_merged_figure(batch_groups, 'batch_size', metric, fixed_params_list)
        merged_figures['batch_size'].append(fig_path)
        print(f"Created merged figure: {fig_path}")
    
    # Create multi-metric merged figure
    fig_path = create_merged_multi_metric_figure(batch_groups, 'batch_size', fixed_params_list)
    merged_figures['batch_size'].append(fig_path)
    print(f"Created merged multi-metric figure: {fig_path}")

# -------------------------------------------------------------------------
# 4.4 Merged comparisons for Learning Rate
# -------------------------------------------------------------------------
if len(lr_groups) > 1:
    print("\nCreating merged figures for Learning Rate...")
    
    # Get fixed parameters for each group
    fixed_params_list = []
    for group in lr_groups:
        fixed_params = [p for p in parameters if p != 'lr' and group[p].nunique() == 1]
        fixed_params_list.append(fixed_params)
    
    # Create merged figures
    for metric in metrics:
        fig_path = create_merged_figure(lr_groups, 'lr', metric, fixed_params_list)
        merged_figures['lr'].append(fig_path)
        print(f"Created merged figure: {fig_path}")
    
    # Create multi-metric merged figure
    fig_path = create_merged_multi_metric_figure(lr_groups, 'lr', fixed_params_list)
    merged_figures['lr'].append(fig_path)
    print(f"Created merged multi-metric figure: {fig_path}")

# -------------------------------------------------------------------------
# 4.5 Special case: Compare models with same parameters
# -------------------------------------------------------------------------
print("\nCreating model comparison figures...")

# Group by all parameters except model
model_comparison_params = [p for p in parameters if p != 'model']
model_groups = df.groupby(model_comparison_params)

# Find groups with multiple models
model_comparison_groups = []
fixed_params_list = []
for name, group in model_groups:
    if len(group) > 1 and group['model'].nunique() > 1:
        model_comparison_groups.append(group)
        fixed_params_list.append(model_comparison_params)

# Create model comparison figures
if model_comparison_groups:
    for i, group in enumerate(model_comparison_groups):
        # Create a multi-metric figure comparing models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get fixed parameters for this group
        fixed_params = fixed_params_list[i]
        
        # Create a label from the fixed parameters
        label_parts = []
        for p in fixed_params:
            val = group[p].iloc[0]
            # Format the value (shorten if it's a float)
            if isinstance(val, float) and val != int(val):
                val_str = f"{val:.3f}".rstrip('0').rstrip('.')
            else:
                val_str = str(val)
            label_parts.append(f"{p.replace('_', ' ')}={val_str}")
        
        # Plot each model
        for model, model_group in group.groupby('model'):
            color = model_colors.get(model, '#333333')
            
            # Plot accuracy
            ax1.bar(model, model_group['final_acc'].mean(), color=color, alpha=0.7)
            ax1.text(model, model_group['final_acc'].mean(), f"{model_group['final_acc'].mean():.2f}", 
                    ha='center', va='bottom', fontsize=10)
            
            # Plot training time
            ax2.bar(model, model_group['elapsed_time'].mean(), color=color, alpha=0.7)
            ax2.text(model, model_group['elapsed_time'].mean(), f"{model_group['elapsed_time'].mean():.0f}s", 
                    ha='center', va='bottom', fontsize=10)
        
        # Set labels and titles
        ax1.set_ylabel('Final Accuracy (%)')
        ax1.set_title('Accuracy by Model')
        
        ax2.set_ylabel('Training Time (s)')
        ax2.set_title('Training Time by Model')
        
        # Add fixed parameters as text annotation
        fixed_text = "\n".join(label_parts)
        fig.text(0.01, 0.02, fixed_text, fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
        
        # Set the figure title
        fig.suptitle(f'Model Comparison (Group {i+1})', fontsize=14)
        
        # Save the figure
        filename = f"model_comparison_group{i+1}.png"
        fig_path = os.path.join(merged_dir, filename)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created model comparison figure: {fig_path}")

# -------------------------------------------------------------------------
# 5. Summary of created figures
# -------------------------------------------------------------------------
print(f"\nDone! All localized figures saved to: {localized_dir}")
print(f"Total localized figures created: {total_figures}")

print(f"\nMerged comparison figures saved to: {merged_dir}")
total_merged = sum(len(figures) for figures in merged_figures.values())
print(f"Total merged figures created: {total_merged}")

# Print summary of figures created for each parameter
print("\nLocalized figures by parameter:")
for param, figures in created_figures.items():
    if figures:
        print(f"  - {param}: {len(figures)} figures")
        
print("\nMerged figures by parameter:")
for param, figures in merged_figures.items():
    if figures:
        print(f"  - {param}: {len(figures)} figures")


