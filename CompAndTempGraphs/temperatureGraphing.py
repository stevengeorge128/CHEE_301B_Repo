import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_excel("301B_Temp_Comp_Data.xlsx", sheet_name="Sheet1")


# Get x values
x = df["Tray"]

for col in df.columns[1:]:
    plt.figure()
    plt.plot(x, df[col])
    plt.xticks(np.arange(1, df["Tray"].max()+1))
    plt.yticks(np.arange(76, 84 + 0.5, 0.5))
    plt.grid()
    plt.title(f"{col}")
    plt.xlabel("Tray Number")
    plt.ylabel("Temperature (°C)")
    plt.savefig(os.path.join("./", f"temp_{col}.png"), dpi=300, bbox_inches="tight")

# plt.show()