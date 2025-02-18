import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



df = pd.read_excel("301B_Temp_Comp_Data.xlsx", sheet_name="Sheet2")
print(df.head())


feed_trays=	[-1, 4, 1, 8, 4, 5, 4]
# Get x values
x = df["Tray"]

i = 0
for col in df.columns[1:]:
    plt.figure()
    plt.plot(x, df[col])
    plt.xticks(np.arange(1, df["Tray"].max()+1))
    plt.yticks(np.arange(0, 1 + 0.1, 0.1))
    plt.grid()
    plt.title(f"{col}")
    plt.xlabel("Tray Number")
    plt.ylabel("Mole Fraction Ethanol (x)")
    plt.savefig(os.path.join("./", f"comp_{col}.png"), dpi=300, bbox_inches="tight")

    
    i += 1
    # break
# plt.show()