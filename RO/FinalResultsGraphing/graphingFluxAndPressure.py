import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("RO_Worksheet.xlsx", sheet_name="fluxVsPressure")
unique_runs = df['Run'].unique()

for run in unique_runs:
    run_df = df[df['Run'] == run]
    

    y = df["Transmembrane Pressure drop, kPa"]
    x = df["Water Flux corrected to 25°C Jw, L/m^2-hr"]

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.grid(zorder=1)
    plt.ylabel("Water Flux corrected to 25°C Jw, L/m^2-hr")
    plt.xlabel("Transmembrane Pressure drop, kPa)")

    # plt.text(2.5,2,equation_text)

    # Scatter plots for individual trials
    # plt.scatter(x,y, color=
    #             "red", marker="o", label=f"Experiment {run}", s=15, zorder=5)
    # plt.scatter(x_to_graph, df["Conductivity_C_(mS)"], color="blue", label="Both Runs", s=15, zorder=5)
    
    plt.scatter(run_df['Water Flux corrected to 25°C Jw, L/m^2-hr'],
            run_df['Transmembrane Pressure drop, psi'], 
            marker='o',
            label=f"Experiment {run}" )

    # Display grid and legend
    plt.grid(zorder=1)
    plt.legend()

    # Show plot
    plt.show()