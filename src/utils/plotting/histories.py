"""
Plots for training / evaluation histories.
--------------------------------------------------------------------------------
`src.utils.plotting.histories`

"""
import seaborn           as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


# ================================================================================
# Plot training and validation metrics for multiple models
# ================================================================================
def plot_model_histories(
        histories      : dict,                     # History objects from the training loop
        ignore_first_n : int = 2,                  # Leave the first N epochs out of the plots (high losses)
        FONT           : str = "Times New Roman",  # Font of plot text
):
    
    # --------------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------------
    # Metrics tracked in the run_epoch function
    metrics = ["epoch_loss", "box_loss", "win_loss", "win_acc", "win_mse"]
    titles  = [
        "Total Epoch Loss", 
        "Box Score Loss", 
        "Win Classification Loss", 
        "Win Accuracy", 
        "Win MSE (Brier Score)"
    ]
    
    # Set up a grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
    axes = axes.flatten()
    
    # Generate a dark color palette for clear contrast
    colors = sns.color_palette("bright", n_colors=len(histories))
    
    # ================================================================================
    # Each panel 
    # ================================================================================
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        for (model_name, history), color in zip(histories.items(), colors):
            
            # --------------------------------------------------------------------------------
            # 1) Filter out the first N epochs (some runs start with very high losses)
            # --------------------------------------------------------------------------------
            filtered_history = [h for h in history if h["epoch"] >= ignore_first_n]
            if not filtered_history: continue
            
            # Extract training metrics
            epochs     = [h["epoch"]             for h in filtered_history]
            train_vals = [h["train"].get(metric) for h in filtered_history]
            
            # --------------------------------------------------------------------------------
            # 2) Training Line
            # --------------------------------------------------------------------------------
            ax.plot(epochs, train_vals, 
                linestyle = "-", 
                alpha     = 0.4, 
                linewidth = 1, 
                color     = color, 
                label     = f"{model_name}"
            )
            
            # --------------------------------------------------------------------------------
            # 3) Validation Line
            # --------------------------------------------------------------------------------
            # No label here, just the dotted version of the training line
            if any(h.get("val") for h in filtered_history):
                val_vals = [h["val"].get(metric) for h in filtered_history]
                ax.plot(epochs, val_vals, 
                    linestyle = "--", 
                    alpha     = 1.0, 
                    linewidth = 1.5, 
                    color     = color, 
                    # label=f"{model_name} (Val)"
                )
                
        # --------------------------------------------------------------------------------
        # Plot Style/Formatting
        # --------------------------------------------------------------------------------
        ax.set_title(title, fontsize=14) # fontweight="bold"
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
    # ================================================================================
    # Use the 6th subplot for the legend
    # ================================================================================
    # Hide the empty 6th subplot's graph elements
    axes[-1].axis("off")

    # Pull the color labels from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()

    # Add proxy lines to explain the line styles (Train vs Validation)
    train_proxy = Line2D([0], [0], color="grey", linestyle="-",  alpha=0.4, linewidth=2.0)
    val_proxy   = Line2D([0], [0], color="grey", linestyle="--", alpha=1.0, linewidth=2.5)

    # Append the proxies to our legend lists
    handles.extend([train_proxy, val_proxy])
    labels .extend(["Training", "Validation"])

    # Place the legend in the center of the empty plot
    axes[-1].legend(handles, labels, loc="center", fontsize=14, frameon=False)

    plt.tight_layout()
    plt.show()




