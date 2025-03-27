import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_excel("RO_Worksheet.xlsx", sheet_name="recoveryVsPressure")
unique_runs = df['Run'].unique()

# Create figure
plt.figure(figsize=(8, 8))
plt.grid(zorder=1)
plt.ylabel(r"Water Percent Recovery", fontsize=20, fontname="Arial")
plt.xlabel(r"Transmembrane Pressure Drop (kPa)", fontsize=20, fontname="Arial")
plt.tick_params(axis='both', which='major', labelsize=16)

# Plot each run
for run in unique_runs:
    run_df = df[df['Run'] == run]
    
    x = pd.to_numeric(run_df["Transmembrane Pressure drop, kPa"], errors='coerce')
    y = pd.to_numeric(run_df["%"], errors='coerce')
    
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]
    
    # Assign salt and shape
    if run in [1, 2]:
        salt = "NaCl"
        # marker = 'o'  # Circle for NaCl
        if run == 1:
            marker = "o"
            color = "red"
        else:
            marker = "x"
            color = "orange"
        zorder=5
    else:
        salt = r"$MgCl_{2}$"
        # marker = '+'  # Square for MgCl2
        if run == 3:
            marker = "s"
            color = "blue"
        else:
            color = "green"
            marker = "^"
        zorder=4

        
    concentration = "1 g/L" if run in [1, 3] else "5 g/L"
    label = f"{concentration} {salt}"

    # Plot
    plt.scatter(x, y, s=100, marker=marker, label=label, zorder=zorder, color=color)

# Style
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)

plt.legend(fontsize=14, loc="best")
plt.tight_layout()
plt.savefig("Combined_RecoveryVsPressure_Shapes.png", dpi=150)
# plt.show()
