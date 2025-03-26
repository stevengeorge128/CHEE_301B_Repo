import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("RO_Worksheet.xlsx", sheet_name="PDiffWithOsmoticPressureDiff")

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

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.grid(zorder=1)
    plt.ylabel("Water Flux corrected to 25°C Jw, L/m^2-hr",fontsize=16, fontname="Arial")
    plt.xlabel("Delta P - Delta Pi (kPa)",fontsize=16, fontname="Arial")
    
    y = run_df["Water Flux corrected to 25°C Jw, L/m^2-hr"]
    dP = run_df['Transmembrane Pressure drop, kPa']
    dPdPi = run_df['Delta P - Delta Pi (kPa)']
    plt.scatter(dP,
            y, 
            marker='o',
            s=50,
            label=f"Experiment {run}",
            )
    
    plt.scatter(dPdPi,
            y, 
            marker='o',
            s=50,
            label=f"Experiment {run}",
            )
    
    
    deg = 1 # Degree of the polynomial, can adjust to 2, 4, etc.

    fluxOfdP = np.polyfit(dP, y, deg)
    fluxOfdPdPi
    fx = np.poly1d(FluxOfP)
    x_vals = np.linspace(min(x), max(x), 1000)
    plt.plot(x_vals, fx(x_vals), label=poly_to_string(fx, "Flux", "(dP-dPi)"), linestyle="--")


    
    
    ax = plt.gca()  # Get current axis

    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Display grid and legend
    plt.grid(zorder=1)
    plt.legend(fontsize="16",loc="upper left")


    # Show plot
    plt.show()
    # plt.savefig(f"FluxVsPDiff{run}")