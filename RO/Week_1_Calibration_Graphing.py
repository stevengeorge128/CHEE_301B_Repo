import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import statsmodels.api as sm

def main():
    df = pd.read_excel("RO_Week_1_Data.xlsx", sheet_name="Results")
    # print(df.head())
    # Get x values
    x_original = df["Concentration_(g/L)"]
    y1 = df["Conductivity_A_(mS)"]
    y2 = df["Conductivity_B_(mS)"]
    y3 = df["Conductivity_C_(mS)"]
    
    all_y = np.concatenate([y1,y2,y3])
    all_x = np.tile(x_original,3)
    # print(all_y)
    # print(all_x)

    
    x = df["Concentration_(g/L)"]
    y = pd.concat([df["Conductivity_A_(mS)"], df["Conductivity_B_(mS)"], df["Conductivity_C_(mS)"]])
    x = sm.add_constant(np.tile(x, 3))
    model = sm.OLS(y, x).fit()
    def equation(x):
        m = model.params[1]
        b = model.params[0]
        return m * x + b
    
    print(model.summary())
    conf_int = model.conf_int(alpha=0.05)
    print("Confidence interval")
    # print(f"p is {p}")
    # print(conf_int)
    print("95% Confidence Intervals for Regression Coefficients:")
    print(f"Intercept: {conf_int.loc['const', 0]:.5f} to {conf_int.loc['const', 1]:.5f}")
    print(f"Slope: {conf_int.loc['x1', 0]:.5f} to {conf_int.loc['x1', 1]:.5f}")
    
    plt.figure(figsize=(8,8)) 
    plt.grid(zorder=1)
    plt.xlabel("Concentration (g/L)")
    plt.ylabel("Conductivity (mS)")
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)
    
    def equation2(x):
        return slope * x + intercept
    
    x_to_graph = df["Concentration_(g/L)"]
    x_linespace =np.linspace(x_to_graph.min(), x_to_graph.max(), 10000)

    plt.scatter(x_to_graph, df["Conductivity_A_(mS)"],color="black",marker="x", label="Trial A",s=15,zorder=5)
    plt.scatter(x_to_graph, df["Conductivity_B_(mS)"],color="red",marker="o", label="Trial B",s=15,zorder=5)
    plt.scatter(x_to_graph, df["Conductivity_C_(mS)"],color="blue", label="Both Runs",s=15,zorder=5)
    plt.plot(x_linespace, equation(x_linespace), color = "blue", label="statsmodel", zorder =2)
    plt.plot(x_linespace, equation2(x_linespace),linestyle="--" , color="red", label="scipy", zorder = 3)

    plt.grid(zorder=1)
    plt.legend()

    plt.show()
    
main()