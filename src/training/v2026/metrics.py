"""
Log metric progress during the training loop 
--------------------------------------------------------------------------------
`src.training.v2026.metrics`

"""
from ...utils.logging import RESET, BOLD, UNBOLD, BRIGHT_GREY, HLINE

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------
# Highlight different metrics with different formatting
HIGHLIGHTS_DICT = {
    # Game source-based highlighting
    "rs": f"", # Regular season
    "tr": f"", # NCAA tournament
    "st": f"", # Secondary tournaments

    # Metric-based highlighting
    "win_mse": f"",
}

# Metrics we expect
ORDERED_KEYS   = ["epoch_loss", "box_loss", "win_loss", "win_acc", "win_mse"]
ORDERED_RENAME = {"epoch_loss": "loss"}


# --------------------------------------------------------------------------------
# Simple metric printing
# --------------------------------------------------------------------------------
def _format_metrics(metrics: dict, m_type: str = "") -> str:
    if not metrics: return "None"

    # Highlight based on the type of metrics we have
    highlight = HIGHLIGHTS_DICT.get(m_type, "")

    # Get all metrics we expect to have
    parts = []
    for key in ORDERED_KEYS:
        if key in metrics: 
            key_str = ORDERED_RENAME.get(key, key)
            parts.append(f"{key_str}={metrics[key]:7.4f}")

    # Include any extra keys too
    for key, value in metrics.items():
        if key not in ORDERED_KEYS:
            if isinstance(value, (int, float)): parts.append(f"{key}={value:7.4f}")
            else:                               parts.append(f"{key}={value    }")

    return " | ".join(parts)


# ================================================================================
# Print a summary of the metrics after a single epoch of training
# ================================================================================
def print_epoch_summary(
        epoch             : int,                 # Current epoch
        num_epochs        : int,                 # Total training epochs
        train_metrics     : dict,                # Regular season metrics
        secondary_metrics : dict | None = None,  # Secondary tournament metrics
        val_metrics       : dict | None = None,  # NCAA tournament metrics 
) -> None:
    # Epoch progress
    print(f"\nEpoch {epoch}/{num_epochs - 1}")

    # Training progress
    print(f"  Train     : {_format_metrics(train_metrics)}")

    # NCAA tournament progress
    if (val_metrics is not None) and (len(val_metrics) > 0):
        print(f"  NCAA      : {_format_metrics(val_metrics)}")

    # Secondary tournament progress
    if (secondary_metrics is not None) and (len(secondary_metrics) > 0):
        print(f"{BRIGHT_GREY}  Secondary : {_format_metrics(secondary_metrics)}{RESET}")


# --------------------------------------------------------------------------------
# Get metrics for the best performing epoch
# --------------------------------------------------------------------------------
# Finds the dictionary in the list where history["val"]["win_mse"] is lowest
def _get_best_epoch(history):
    best_entry = min(history, key=lambda x: x["val"]["win_mse"])
    return best_entry

def print_best_epoch(history, num_epochs: int = 0):
    best_entry = _get_best_epoch(history)
    
    print(f"\n{HLINE}\n{BOLD}Best Epoch:{UNBOLD}\n{HLINE}")

    print_epoch_summary(
        epoch             = best_entry["epoch"],
        num_epochs        = num_epochs,
        train_metrics     = best_entry["train"    ],
        secondary_metrics = best_entry["secondary"],
        val_metrics       = best_entry["val"      ],
    )
