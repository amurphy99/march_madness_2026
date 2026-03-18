"""
Utility class to help track training progress
--------------------------------------------------------------------------------
`src.training.utils.average_meter`

Should help with tracking running metrics in the training loop.

"""

# --------------------------------------------------------------------------------
# Calculates and stores the average and current value
# --------------------------------------------------------------------------------
class AverageMeter:
    # Create a new instance for each thing I need to track
    def __init__(self):
        self.reset()

    # Reset everything to 0
    def reset(self):
        self.val = self.sum = self.count = self.avg = 0

    # Update tracking
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n

        self.avg = self.sum / self.count
