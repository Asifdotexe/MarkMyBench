import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Objective Function: Hypothetical search space
def objective_function(x, y):
    return -np.sin(x) * np.cos(y) + 0.1 * (x ** 2 + y ** 2)

# Create a 2D grid for visualization
x = np.linspace(-5, 5, 50)  # Increased grid resolution
y = np.linspace(-5, 5, 50)  # Increased grid resolution
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Setup plot
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Comparison of Optimization Methods by Asif Sayyed", fontsize=14, weight="bold")
ax.set_xlabel("Hyperparameter 1", fontsize=12)
ax.set_ylabel("Hyperparameter 2", fontsize=12)

# Scatter plots for each optimization method (different colors and shapes)
scatter_grid = ax.scatter([], [], c="blue", s=40, alpha=0.5, marker='o', label="GridSearchCV")  # Circle
scatter_random = ax.scatter([], [], c="orange", s=40, alpha=0.5, marker='s', label="RandomizedSearchCV")  # Square
scatter_bayesian = ax.scatter([], [], c="green", s=40, alpha=0.5, marker='^', label="Bayesian Optimization")  # Triangle

# Add a circle at the center for optimal parameter region
circle = plt.Circle((0, 0), 1.5, color='red', fill=False, linewidth=2, label="Optimal Region", linestyle=':')
ax.add_artist(circle)

# Add a red dot at the center to represent the most optimal parameter (0, 0)
ax.scatter(0, 0, c='red', alpha=0.5, s=100, edgecolors='black', label="Most Optimal Parameter", zorder=5)

# Data for GridSearchCV (60 points)
grid_points = []
rows = 7  # Reduced number of rows
cols = 7  # Reduced number of columns

# Distribute points in a matrix-like grid across the plot for GridSearchCV
for i in range(rows):
    for j in range(cols):
        x_val = -5 + (j * (10 / (cols - 1)))  # Spacing x-axis points evenly
        y_val = -5 + (i * (10 / (rows - 1)))  # Spacing y-axis points evenly
        grid_points.append([x_val, y_val])

# Data for RandomizedSearchCV (60 points) - Different random seed
num_points = 60  # Adjusted to 60 points
random_sampled_points = []
np.random.seed(43)  # Different seed for RandomizedSearchCV

# Random points scattered more naturally across the search space
for _ in range(num_points):
    random_sampled_points.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])

# Data for Bayesian Optimization (60 points) - Different random seed
np.random.seed(44)  # Different seed for Bayesian Optimization
bayesian_points = []
for i in range(60):  # Adjusted to 60 iterations
    # As the search progresses, the points become focused towards a central region
    if i < 30:
        # Random sampling initially
        point = [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
    else:
        # Gradually focus the search in the vicinity of the best-performing points
        point = [np.random.normal(0, 2), np.random.normal(0, 2)]  # Centered sampling

    bayesian_points.append(point)

# Counter for the number of iterations
iteration_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, fontsize=12, color='black', ha='center', va='bottom')

# Update function for all three methods
def update(frame):
    # Update GridSearchCV
    if frame < len(grid_points):
        scatter_grid.set_offsets(grid_points[:frame + 1])

    # Update RandomizedSearchCV
    if frame < len(random_sampled_points):
        scatter_random.set_offsets(random_sampled_points[:frame + 1])

    # Update Bayesian Optimization
    if frame < len(bayesian_points):
        scatter_bayesian.set_offsets(bayesian_points[:frame + 1])

    # Update iteration counter
    iteration_text.set_text(f"Iteration: {frame + 1}/{max(len(grid_points), len(random_sampled_points), len(bayesian_points))}")

    return scatter_grid, scatter_random, scatter_bayesian, iteration_text


# Update function for all three methods
def update(frame):
    # Update GridSearchCV
    if frame < len(grid_points):
        scatter_grid.set_offsets(grid_points[:frame + 1])

    # Update RandomizedSearchCV
    if frame < len(random_sampled_points):
        scatter_random.set_offsets(random_sampled_points[:frame + 1])

    # Update Bayesian Optimization
    if frame < len(bayesian_points):
        scatter_bayesian.set_offsets(bayesian_points[:frame + 1])

    # Update iteration counter
    iteration_text.set_text(f"Iteration: {frame + 1}/{max(len(grid_points), len(random_sampled_points), len(bayesian_points))}")

    return scatter_grid, scatter_random, scatter_bayesian, iteration_text

# Animations for all three search methods
anim = FuncAnimation(fig, update, frames=max(len(grid_points), len(random_sampled_points), len(bayesian_points)),
                     interval=200, blit=True)

# Adjust layout and make the legend translucent
plt.tight_layout()
plt.legend(loc="lower right", frameon=True, facecolor='white')  # Solid white background for the legend
anim.save('optimization_animation.gif', writer='pillow', fps=1)
plt.show()
