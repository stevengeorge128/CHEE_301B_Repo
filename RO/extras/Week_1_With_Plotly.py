import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import plotly.graph_objects as go

def main():
    # Load data from Excel
    df = pd.read_excel("RO_Week_1_Data.xlsx", sheet_name="Results")

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
    x = sm.add_constant(all_x)  # Add intercept term
    y = pd.concat([y1, y2, y3]).reset_index(drop=True)  # Flatten y values

    # Fit Ordinary Least Squares (OLS) model
    model = sm.OLS(y, x).fit()

    # Define function for the statsmodels regression equation (y = mx + b)
    def equation(x):
        m = model.params.iloc[1]  # Slope
        b = model.params.iloc[0]  # Intercept
        return m * x + b

    # Print model summary
    print(model.summary())

    # Compute and print 95% confidence intervals
    conf_int = model.conf_int(alpha=0.05)
    print("95% Confidence Intervals for Regression Coefficients:")
    print(f"Intercept: {conf_int.iloc[0, 0]:.5f} to {conf_int.iloc[0, 1]:.5f}")
    print(f"Slope: {conf_int.iloc[1, 0]:.5f} to {conf_int.iloc[1, 1]:.5f}")

    # Compute linear regression using SciPy (for comparison)
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)

    # Define function for SciPy regression equation
    def equation2(x):
        return slope * x + intercept

    # Generate x values for plotting regression lines
    x_linespace = np.linspace(x_original.min(), x_original.max(), 10000)  # Smooth fit

    # Create Plotly figure
    fig = go.Figure()

    # Scatter plots for individual trials
    fig.add_trace(go.Scatter(x=x_original, y=y1, mode='markers', marker=dict(color='black', symbol='x', size=6),
                             name="Trial A"))
    fig.add_trace(go.Scatter(x=x_original, y=y2, mode='markers', marker=dict(color='red', symbol='circle', size=6),
                             name="Trial B"))
    fig.add_trace(go.Scatter(x=x_original, y=y3, mode='markers', marker=dict(color='blue', size=6),
                             name="Trial C"))

    # Plot regression lines
    fig.add_trace(go.Scatter(x=x_linespace, y=equation(x_linespace), mode='lines', line=dict(color='blue'),
                             name="Statsmodels Fit"))
    fig.add_trace(go.Scatter(x=x_linespace, y=equation2(x_linespace), mode='lines', line=dict(color='red', dash='dash'),
                             name="SciPy Fit"))

    # Layout customization
    fig.update_layout(
        title="Regression Analysis",
        xaxis_title="Concentration (g/L)",
        yaxis_title="Conductivity (mS)",
        legend=dict(x=0, y=1),
        template="plotly_white"
    )

    # Show interactive plot
    fig.show()

main()
