# march_madness_2026

Hopefully in future years I can just make forks from this repository


## Project Architecture
```diff
 march_madness_2026/
 ├── data/..
 ├── notebooks/..
 └── src/
     ├── models/
     │    ├── model_2025.py    # Model architecture I used in 2025
     │    └── ...              # New model architectures I am experimenting with
     │
     ├── processing/
     │   ├── features/..      # Logic for individual metrics
     │   └── datasets/        # Logic for the final data structure
     │      ├── history.py    # Builds game histories for each team
     │      ├── generator.py  # Create X, Y data for model training
     │      └── ...
     │
     └── ...


```

