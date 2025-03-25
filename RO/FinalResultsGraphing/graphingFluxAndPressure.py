import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("RO_Worksheet.xlsx", sheet_name="fluxVsPressure")
unique_runs = df['Run'].unique()

for run in unique_runs:
    run_df = df[df['Run'] == run]


    # Create plot
    plt.figure(figsize=(8, 8))
    plt.grid(zorder=1)
    plt.ylabel("Water Flux corrected to 25°C Jw, L/m^2-hr",fontsize=16, fontname="Arial")
    plt.xlabel("Transmembrane Pressure drop, kPa",fontsize=16, fontname="Arial")

    
    plt.scatter(run_df['Water Flux corrected to 25°C Jw, L/m^2-hr'],
            run_df['Transmembrane Pressure drop, kPa'], 
            marker='o',
            s=50,
            label=f"Experiment {run}",
            )
    ax = plt.gca()  # Get current axis

    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Display grid and legend
    plt.grid(zorder=1)
    plt.legend(fontsize="16",loc="upper left")


    # Show plot
    # plt.show()
    plt.savefig(f"FluxVsP{run}")