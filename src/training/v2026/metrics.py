"""
Log metric progress during the training loop 
--------------------------------------------------------------------------------
`src.training.v2026.metrics`

"""
from ...utils.logging import RESET, BRIGHT_GREY


# Highlight different metrics with different formatting
HIGHLIGHTS_DICT = {
    # Game source-based highlighting
    "rs": f"", # Regular season
    "tr": f"", # NCAA tournament
    "st": f"", # Secondary tournaments

    # Metric-based highlighting
    "win_mse": f"",
}

# --------------------------------------------------------------------------------
# Simple metric printing
# --------------------------------------------------------------------------------
def _format_metrics(metrics: dict, m_type: str = "") -> str:
    if not metrics: return "None"

    # Highlight based on the type of metrics we have
    highlight = HIGHLIGHTS_DICT.get(m_type, "")

    # Metrics we expect
    ordered_keys = ["epoch_loss", "box_loss", "win_loss", "win_acc", "win_mse"]
    parts        = []

    for key in ordered_keys:
        if key in metrics: 
            parts.append(f"{key}={metrics[key]:7.4f}")

    # Include any extra keys too
    for key, value in metrics.items():
        if key not in ordered_keys:
            if isinstance(value, (int, float)): parts.append(f"{key}={value:7.4f}")
            else:                               parts.append(f"{key}={value    }")

    return " | ".join(parts)


# ================================================================================
# Model Definition
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


