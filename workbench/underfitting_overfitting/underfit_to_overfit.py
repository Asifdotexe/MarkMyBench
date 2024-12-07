import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Generate synthetic data
np.random.seed(42)
X_train = np.random.uniform(-3, 3, 50).reshape(-1, 1)  # Training features
y_train = X_train**3 - 3 * X_train + 5 + np.random.normal(0, 3, X_train.shape[0]).reshape(-1, 1)  # Training target

# Test data for model predictions
X_test = np.linspace(-3, 3, 300).reshape(-1, 1)

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(10, 7))

# Define polynomial degrees for animation frames
polynomial_degrees = np.linspace(1, 20, 100).astype(int)
polynomial_degrees = np.append(polynomial_degrees, [20] * 20)  # Add pause at the final degree


def animate(frame_idx):
    """
    Update the plot for a given frame in the animation.

    :param frame_idx: int
        The index of the current frame in the animation sequence.

    This function performs the following:
    - Clears the existing plot.
    - Computes polynomial features for the specified degree.
    - Fits a linear regression model to the training data.
    - Predicts values for the test dataset.
    - Calculates the training error (MSE).
    - Updates the plot with data points, model predictions, and relevant labels.

    Additional Notes:
    - Degrees less than 3 are labeled as "Underfitting."
    - Degrees greater than 15 are labeled as "Overfitting."
    - Degrees in the range of 3-15 are labeled as "Optimal Fit."
    """
    ax.clear()
    degree = polynomial_degrees[frame_idx]

    # Create polynomial features
    poly_features = PolynomialFeatures(degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Fit the regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_test_pred = model.predict(X_test_poly)

    # Calculate training error
    train_error = mean_squared_error(y_train, model.predict(X_train_poly))

    # Plot the data and model predictions
    ax.scatter(
        X_train,
        y_train,
        color="#1f77b4",
        label="Data",
        alpha=0.8,
        edgecolor="black",
    )
    ax.plot(
        X_test,
        y_test_pred,
        color="#d62728",
        linewidth=2,
        label=f"Degree: {degree}\nTrain Error: {train_error:.2f}",
    )

    # Customize plot appearance
    ax.legend(fontsize=12, loc="lower right", frameon=True, shadow=True)
    ax.set_title("Underfitting to Overfitting", fontsize=16, fontweight="bold")
    ax.set_xlabel("X", fontsize=14)  # Updated X-axis label
    ax.set_ylabel("y", fontsize=14)  # Updated Y-axis label
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_ylim(y_train.min() - 5, y_train.max() + 5)

    # Annotate underfitting, optimal fit, and overfitting regions
    if degree < 3:
        ax.text(-2.5, y_train.max() - 2, "Underfitting", fontsize=14, color="orange", weight="bold")
    elif degree > 15:
        ax.text(-2.5, y_train.max() - 2, "Overfitting", fontsize=14, color="orange", weight="bold")
    else:
        ax.text(-2.5, y_train.max() - 2, "Optimal Fit", fontsize=14, color="green", weight="bold")

    # Add author label
    ax.text(2.5, y_train.min() - 8, "Created by Asif Sayyed", fontsize=10, color="gray", ha="right", style="italic")


# Create and save the animation
animation = FuncAnimation(fig, animate, frames=len(polynomial_degrees), interval=100)
animation.save("underfitting_overfitting_optimized.gif", writer="pillow", fps=20)
plt.show()
