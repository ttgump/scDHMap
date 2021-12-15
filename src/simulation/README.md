# Code for generating simulated datasets

splatter_simulation_dropout.R - Simulate datasets with various dropout rates

splatter_simulation_batch.R - Simulate datasets with batch effects

splatter_simulation_three_branches.R - Simulate datasets with three branches used for trajectory interpretation and denoising counts

run_transform_paths.py - Using Poincare embedding for inference poincare pseudotime of cells in each branch (The three-branch datasets). First transform the Poincare origin to the starting point of each branch. One branch could contain multiple points with the first step, we use hyperbolic centroid of these points as the starting point. Poincare pseudotime is infered as the Poincare distance between points to the starting point.