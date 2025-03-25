import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("RO_Worksheet.xlsx", sheet_name="cVsCDrop")
unique_runs = df['Run'].unique()

def poly_to_string(poly, yStr, xStr):
    terms = []
    deg = len(poly.coefficients) - 1
    for i, coef in enumerate(poly.coefficients):
        power = deg - i
        if abs(coef) < 1e-8:  # Skip tiny coefficients
            continue
        if power == 0:
            terms.append(f"{coef:.4f}")
        elif power == 1:
            terms.append(f"{coef:.4f}·{xStr}")
        else:
            terms.append(f"{coef:.4f}·{xStr}^{power}")
    result = " + ".join(terms)
    return yStr + " = " + result

for run in unique_runs:
    run_df = df[df['Run'] == run]
    # print("Run df is ")
    # print(run_df)

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.grid(zorder=1)
    plt.ylabel("Salt Flux corrected to 25°C Js, mol/m^2-hr",fontsize=16, fontname="Arial")
    plt.xlabel("Salt concentration Gradient, DCs, mol/L",fontsize=16, fontname="Arial")
    
    y = run_df["Salt Flux corrected to 25°C Js, mol/m^2-hr"]
    x = run_df["Salt concentration Gradient, DCs, mol/L"]
    plt.scatter(x,
            y, 
            marker='o',
            s=50,
            label=f"Experiment {run}",
            )
    
    
    deg = 1 # Degree of the polynomial, can adjust to 2, 4, etc.
    print(x)
    print(y)
    print(deg)
    FluxOfP = np.polyfit(x, y, deg)
    fx = np.poly1d(FluxOfP)
    x_vals = np.linspace(min(x), max(x), 1000)
    plt.plot(x_vals, fx(x_vals), label=poly_to_string(fx, "Flux", "dC"), linestyle="--")


    
    
    ax = plt.gca()  # Get current axis

    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Display grid and legend
    plt.grid(zorder=1)
    plt.legend(fontsize="16",loc="upper left")


    # Show plot
    # plt.show()
    plt.savefig(f"SaltFluxVsCDiff{run}")