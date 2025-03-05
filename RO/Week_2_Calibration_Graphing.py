import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

def main():
    # Load data from Excel
    df = pd.read_excel("RO_Week_2_Data.xlsx", sheet_name="Results")

    # Extract x values (independent variable)
    x_original = df["Concentration_(g/L)"]

    # Extract y values (dependent variables from multiple trials)
    y1 = df["Conductivity_A_(mS)"]
    y2 = df["Conductivity_B_(mS)"]
    y3 = df["Conductivity_C_(mS)"]

    # Combine all y values into one dataset for regression
    all_y = np.concatenate([y1, y2, y3])
    all_x = np.tile(x_original, 3)  # Repeat x-values to match y-values

    # Prepare data for statsmodels regression
    x = np.tile(x_original, 3)  # Ensure same x-values for all trials
    x = sm.add_constant(x)  # Add intercept term
    y = pd.concat([y1, y2, y3])  # Flatten y values

    # Fit Ordinary Least Squares (OLS) model
    model = sm.OLS(y, x).fit()

    # Define function for the statsmodels regression equation (y = mx + b)
    def equation(x):
        m = model.params[1]  # Slope
        b = model.params[0]  # Intercept
        return m * x + b

    # Print model summary
    print(model.summary())

    # Compute and print 95% confidence intervals
    conf_int = model.conf_int(alpha=0.05)
    print("95% Confidence Intervals for Regression Coefficients:")
    print(f"Intercept: {conf_int.loc['const', 0]:.5f} to {conf_int.loc['const', 1]:.5f}")
    print(f"Slope: {conf_int.loc['x1', 0]:.5f} to {conf_int.loc['x1', 1]:.5f}")

    # Compute linear regression using SciPy (for comparison)
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)

    # Define function for SciPy regression equation
    def equation2(x):
        return slope * x + intercept

    # Generate x values for plotting regression lines
    x_to_graph = df["Concentration_(g/L)"]
    x_linespace = np.linspace(x_to_graph.min(), x_to_graph.max(), 10000)  # Smooth fit

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.grid(zorder=1)
    plt.xlabel("Concentration (g/L)")
    plt.ylabel("Conductivity (mS)")

    # Scatter plots for individual trials
    plt.scatter(x_to_graph, df["Conductivity_A_(mS)"], color="black", marker="x", label="Trial A", s=15, zorder=5)
    plt.scatter(x_to_graph, df["Conductivity_B_(mS)"], color="red", marker="o", label="Trial B", s=15, zorder=5)
    plt.scatter(x_to_graph, df["Conductivity_C_(mS)"], color="blue", label="Both Runs", s=15, zorder=5)

    # Plot regression lines
    plt.plot(x_linespace, equation(x_linespace), color="blue", label="Statsmodels Fit", zorder=2)
    plt.plot(x_linespace, equation2(x_linespace), linestyle="--", color="red", label="SciPy Fit", zorder=3)

    # Display grid and legend
    plt.grid(zorder=1)
    plt.legend()

    # Show plot
    plt.show()

main()
