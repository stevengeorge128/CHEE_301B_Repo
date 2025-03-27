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
df = pd.read_excel("RO_Worksheet.xlsx", sheet_name="fluxVsPDiff")
unique_runs = df['Run'].unique()

for run in unique_runs:
    run_df = df[df['Run'] == run]

    # Create plot
    plt.figure(figsize=(10, 10))
    plt.grid(zorder=1)

    plt.ylabel(r"$J_w$ (Water Flux corrected to 25°C), $\frac{L}{m^2 \cdot hr}$", fontsize=24, fontname="Arial")
    plt.xlabel(r"$\Delta P - \Delta \pi$ (kPa)", fontsize=24, fontname="Arial")
    plt.tick_params(axis='both', which='major', labelsize=24)

    y = run_df['Water Flux corrected to 25°C Jw, L/m^2-hr']
    x = run_df['Delta P - Delta Pi (kPa)']
    if (run == 1):
        l = "C = 1 g/L NaCl"
    elif (run == 2):
        l = "C = 5 g/L NaCl"
    elif (run == 3):
        l = r"C = 1 g/L $MgCl_{2}$"
    else:
        l = r"C = 5 g/L $MgCl_{2}$"
    
    plt.scatter(x,
            y, 
            marker='o',
            s=100,
            # label=l,
            color="red",
            zorder=5
            )
    
    plt.scatter([],
        [], 
        marker='o',
        s=100,
        label=l,
        color="white",
        zorder=5
        )
    
    
    deg = 1 # Degree of the polynomial, can adjust to 2, 4, etc.

    FluxOfP = np.polyfit(x, y, deg)
    fx = np.poly1d(FluxOfP)
    x_vals = np.linspace(min(x), max(x), 1000)
    plt.plot(x_vals, fx(x_vals), label=poly_to_string(fx, r"$J_w$", r"($\Delta P - \Delta \pi$)"), lw=5, linestyle="--",zorder=3)
    r2 = calculate_r_squared(x, y, fx)
    
    plt.plot([], [], label=f"R² = {r2:.5f}", color="white")
    # print(fx.coefficients)
    Kw_kPa = fx.coefficients[0]
    if (run < 3):
        Kw_atm = Kw_kPa * 101.325 * 0.99705 * 1000 / 58.11  # 1 atm = 101.325 kPa
    else:
        Kw_atm = Kw_kPa * 101.325 * 0.99705*1000 / 95.211

    # plt.plot([], [], label=f"$K_{{w}}$ = {Kw_kPa:.5f} $\\frac{{L}}{{m^2 \\cdot hr \\cdot kPa}}$", color="white")
    # plt.plot([], [], label=f" = {Kw_atm:.5f} $\\frac{{L}}{{m^2 \\cdot hr \\cdot atm}}$", color="white")
    print("----")
    print(max(x))
    print(min(x))
    print((min(x) + (max(x)-min(x))/2))
    print(fx((min(x) + (max(x)-min(x))/2)))
    # print(fx((max(x)-min(x))/2))
    plt.text((min(x) + (max(x)-min(x))/2), 
             fx((min(x) + (max(x)-min(x))/2)) - 3, 
            f"$K_{{w}}$ = {Kw_kPa:.5f} $\\frac{{L}}{{m^2 \\cdot hr \\cdot kPa}}$\n"
            f"= {Kw_atm:.5f} $\\frac{{mol}}{{m^2 \\cdot hr \\cdot atm}}$",
            # transform=plt.gca().transAxes,  # so it's relative to the axes
            fontsize=24,
            verticalalignment='top',
            # color='white'
            )
    # plt.plot([], [], label=f"$K_{w}$ = {Kw:.5f} $\frac{L}{meters^{2}}$ ")
    # plt.plot([], [], label=f"$K_{{w}}$ = {Kw:.5f} $\\frac{{L}}{{m^2 \\cdot hr \\cdot kPa}}$", color="white")


    
    
    ax = plt.gca()  # Get current axis

    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Display grid and legend
    plt.grid(zorder=1)
    plt.legend(fontsize="24",loc="upper left")


    # Show plot
    # plt.show()
    plt.savefig(f"FluxVsPDiff{run}")