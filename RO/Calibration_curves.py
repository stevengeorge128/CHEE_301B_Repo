import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_calibration(fileName):
    # Load data from Excel
    df = pd.read_excel(fileName, sheet_name="Results")
    
    x_from_excel = df["Concentration_(g/L)"]
    y1 = df["Conductivity_A_(mS)"]
    y2 = df["Conductivity_B_(mS)"]
    y3 = df["Conductivity_C_(mS)"]
    
    # Concat y arrays to get all y data, and replicate x_data 3 times so all 
    # y values have the correct corresponding value in x_data
    y_data = np.concatenate([y1,y2,y3])
    # print(x_from_excel)
    x_data = np.tile(x_from_excel,3)
    
    # print(x_data)
    # print(y_data)
    
    # Add a constant term for the intercept so statsmodel
    # can try and find an intercept
    X = sm.add_constant(x_data)
    # print(x_data)
    # print(X)
    
    # Fit the OLS model to the data 
    model = sm.OLS(y_data, X).fit()
    # print(model.summary())
    
    # Confidence intervals for the model parameters
    conf_intervals = model.conf_int()
    # print(conf_intervals)
    
    # Get prediction results
    pred = model.get_prediction(X)
    pred_summary = pred.summary_frame(alpha=0.05)  # 95% confidence intervals
    # print(pred)
    # Extract confidence intervals
    ci_lower = pred_summary['mean_ci_lower']
    ci_upper = pred_summary['mean_ci_upper']
    print(ci_lower)
    # print(ci_upper)
    
    plt.figure()
    plt.scatter(x_from_excel, y1, marker="x")
    plt.scatter(x_from_excel, y2,marker="*")
    plt.scatter(x_from_excel, y3,marker="+")
    # plt.fill_between(x_from_excel, ci_lower, ci_upper, color='red', alpha=0.3, label='95% CI')
    
    plt.plot(x_data, model.fittedvalues)
    # print(x_from_excel)
    plt.fill_between(x_data, ci_lower, ci_upper, color="red", alpha=0.3, label="95% CI")

    plt.xlabel("Concentration (g/L)")
    plt.ylabel("Conductivity (mS)")
    plt.legend()
    # plt.show()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_from_excel,y=y1))
    
    fig.show()
    
    
    

   

plot_calibration("RO_Week_1_Data.xlsx")
# plot_calibration("RO_Week_2_Data.xlsx")

