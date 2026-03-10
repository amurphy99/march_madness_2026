"""
Training metrics
--------------------------------------------------------------------------------
`src.training.metrics`

"""


# Print header for the metrics
def print_metrics_header():
    print(f"{' '*32} | Regular Season           | Secondary Tournaments    | NCAA Tournament")
    print(f"{'-'*32} | ------------------------ | ------------------------ | ------------------------")

# ================================================================================
# Progress printing
# ================================================================================
def print_train_metrics(epoch, num_epochs, m, validation=False):
    # Printing progress # Box: {m['box_loss']:5.2f},
    loss_str = f"Loss: {m['epoch_loss']:6.4f}, Win: {m['win_loss']:6.4f}"
    prob_str = f"Acc: {m['win_acc']:.4f}, MSE: {m['win_mse']:.4f}"
    
    if validation: print(f"Val => {prob_str}")
    else:          print(f"Epoch [{epoch+1:3} / {num_epochs:3}] | {loss_str} | {prob_str} ")
        

# ================================================================================
# Progress printing
# ================================================================================
def print_train_val_metrics(
        epoch      : int, 
        num_epochs : int, 
        history    : list, 
        t_m, 
        s_m, 
        v_m, 
        *, 
        verbose: int = 1,
):
    # --------------------------------------------------------------------------------
    # Progress strings 
    # --------------------------------------------------------------------------------
    loss_str = f"L: {t_m['epoch_loss']:6.4f}, Win: {t_m['win_loss']:6.4f}" # Box: {m['box_loss']:5.2f},
    prob_str = f"Acc: {t_m['win_acc']:.4f}, MSE: {t_m['win_mse']:.4f}"
    
    # Secondary metrics
    snd_str = f"Acc: {s_m.get('win_acc', 0.5):.4f}, MSE: {s_m.get('win_mse', 0.25):.4f}"
    
    # Validation metrics
    val_str = f"Acc: {v_m['win_acc']:.4f}, MSE: {v_m['win_mse']:.4f}"
    
    # Print progress
    if verbose or ((epoch+0) % 10) == 0 or (epoch+1) == num_epochs:
        print(f"[{epoch+1:3}/{num_epochs:3}] {loss_str} | {prob_str} | {snd_str} | {val_str}")
    
    # --------------------------------------------------------------------------------
    # Save the training history
    # --------------------------------------------------------------------------------
    epoch_history = {"Training": {}, "Secondary": {}, "Validation": {}}
    
    if len(s_m) == 0: save_metrics = [(t_m, "Training"), (v_m, "Validation")]
    else:             save_metrics = [(t_m, "Training"), (s_m, "Secondary"), (v_m, "Validation")]
    
    for metrics, key in save_metrics:
        # Loss (combined, box score only, and win probability)
        epoch_history[key][    "Loss"] = metrics["epoch_loss"]
        epoch_history[key]["Box_Loss"] = metrics[  "box_loss"]
        epoch_history[key]["Win_Loss"] = metrics[  "win_loss"]
        
        # Win probability specific metrics
        epoch_history[key]["Win_Acc"] = metrics["win_acc"]
        epoch_history[key]["Win_MSE"] = metrics["win_mse"]
    
    # Save it
    history.append(epoch_history)

