import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_r_squared(x, y, model):
    y_pred = model(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

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

df = pd.read_excel("RO_Worksheet.xlsx", sheet_name="cVsCDrop")
unique_runs = df['Run'].unique()

for run in unique_runs:
    run_df = df[df['Run'] == run]

    # Convert to numeric and drop rows with NaNs
    x = pd.to_numeric(run_df['Salt concentration Gradient, DCs, mol/L'], errors='coerce')
    y = pd.to_numeric(run_df['Salt Flux corrected to 25°C Js, mol/m^2-hr'], errors='coerce')
    # mask = (~x.isna()) & (~y.isna())
    # x = x[mask]
    # y = y[mask]

    plt.figure(figsize=(12, 12))
    plt.grid(zorder=1)
    plt.ylabel("Js (Salt Flux corrected to 25°C), mol/m^2hr", fontsize=24, fontname="Arial")
    plt.xlabel("Salt Concentration Gradient, mol/L", fontsize=24, fontname="Arial")
    # plt.ylabel(r"$J_s$ (Salt Flux corrected to 25°C), $\frac{mol}{m^2 \cdot hr}$", fontsize=24, fontname="Arial")
    # plt.xlabel(r"$\Delta C$ (Salt Concentration Gradient), $\frac{mol}{L}$", fontsize=24, fontname="Arial")
    plt.tick_params(axis='both', which='major', labelsize=24)

    # Label for each run
    if run == 1:
        l = "C = 1 g/L NaCl"
    elif run == 2:
        l = "C = 5 g/L NaCl"
    elif run == 3:
        l = r"C = 1 g/L $MgCl_{2}$"
    else:
        l = r"C = 5 g/L $MgCl_{2}$"

    # Scatter data
    plt.scatter(x, y, marker='o', s=100, color="red", zorder=5)
    plt.scatter([], [], marker='o', s=100, label=l, color="white", zorder=5)

    # Polynomial fit
    deg = 1
    FluxOfP = np.polyfit(x, y, deg)
    fx = np.poly1d(FluxOfP)
    x_vals = np.linspace(min(x), max(x), 1000)
    plt.plot(x_vals, fx(x_vals), label=poly_to_string(fx, r"$J_s$", r"$\Delta C$"), lw=5, linestyle="--", zorder=3)
    
    r2 = calculate_r_squared(x, y, fx)
    plt.plot([], [], label=f"R² = {r2:.5f}", color="white")

    # Get and convert slope
    Ks = fx.coefficients[0]  # Units: mol/(m²·hr·(mol/L)) = L/(m²·hr)
    
    # Annotate Ks value near center of plot
    x_mid = min(x) + (max(x) - min(x)) / 2
    y_mid = fx(x_mid)
    plt.text(x_mid, y_mid - 3, 
             f"$K_{{s}}$ = {Ks:.5f} $\\frac{{L}}{{m^2 \\cdot hr}}$",
             fontsize=24, verticalalignment='top')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    plt.grid(zorder=1)
    plt.legend(fontsize=24, loc="upper left")
    # plt.tight_layout()
    plt.savefig(f"SaltFlux{run}")
