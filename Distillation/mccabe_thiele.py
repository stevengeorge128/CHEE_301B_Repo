import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

# Data from excel. Easier to copy over than import, etc.
run_names = np.array(["Run 1", "Run 2", "Run 3", "Run 4", "Run 5", "Run 6", "Run 7"])
total = np.array([True, False, False, False, False, False, False])
distillate_mole_fracs = np.array([0.8040763915, 0.7735944349,	0.6492856239, 0.8120534502,	0.7762602948, 0.7768328571, 0.7754975967])
feed_mole_fracs = np.array([0.3032315113, 0.3032315113, 0.309717042, 0.3865057186, 0.38781182 ,0.3865057186, 0.3888187723])
bottoms_mole_fracs = np.array([0.1049134497, 0.1099220247, 0.1280498073, 0.1928839517, 0.2049585951, 0.2116114024, 0.2359981844])

reflux_ratios = np.array([1000000000, 3, 3, 3, 1.5, 10, 3])
feed_temps = np.array([21.1, 21.1, 21.23, 20.14, 20.14, 50.75, 72.73])


def dew_point_curve(x):
    return 38.351*x**3 - 35.618*x**2 - 23.682*x + 97.993

def bubble_point_curve(x):
    return -57.166*x**3 + 111.44*x**2 - 71.469*x + 92.003

def calc_q(dp, bp, f):
    return 1 - 0.5 * (f-0.5*bp)/(dp-0.5*bp)

def y_equals_x(x):
    return x

def el(x):
    return 13.74*(x**5) - 39.452*(x**4) + 42.937*(x**3) - 21.52*(x**2) + 5.2312*x+0.0831

def intersection_el_y(x,y):
    return el(x) - y

def getELx(y):
    return fsolve(intersection_el_y, y, args=(y))
    

def feed_line(x, q, f):
    return ((-q*x)/(1-q)) + f/(1-q)

def rsol(x, R, d):
    return (R * x)/ (R+1 ) + d/(R+1)
   
def intersection_rsol_fl(x,R,d,q,f):
    return rsol(x,R,d) - feed_line(x,q,f)

def ssol(x, x1,y1,x2,y2):
    m = (y2-y1) / (x2-x1)
    b = y1 - m * x1
    return m * x + b

def intersection_fl_el(x, q, f):
    return feed_line(x,q,f) - el(x)

def intersection_fl_rsol(x,q,f,R,d):
    return feed_line(x,q,f) - rsol(x,R,d)

def rsol_prime(R):
    return R/(R+1)

def el_prime(x):
    return 5 * 13.74*(x**4) - 4*39.452*(x**3) + 3*42.937*(x**2) - 2*21.52*(x**1) + 5.2312

def intersection_rsol_el(x,R,d):
    return rsol(x,R,d) - el(x)

def calculate_minimum_reflux(plt, R,d, fl_end,q,f):
    slope = (d-feed_line(fl_end,q,f))/(d-fl_end)
    b = d - slope * d
    xs = [fl_end,d]
    ys = [feed_line(fl_end,q,f), d]
    R_min = slope/(1-slope)
    label = f"R_min = {R_min:.4f}"
    above = False
    k = 0
    if fl_end < d:
        r = np.linspace(fl_end+0.01, d, 1000)
    else:
        r = np.linspace(d, fl_end + 0.01, 10)
    for i in r:
        if (rsol(i,R_min,d) > el(i)):
            above = True
            break
                  
    if not above:
        plt.plot(xs, ys, label=label)
        # print(f", R_min = {R_min:5f}", end="") 
    else:
          
        possible_R_range = np.linspace(1.2, 0.5, 1000)

        i = 0
        real_roots = [0]
        prev_R = 0
        while len(real_roots) < 2 and i < len(possible_R_range):
            R = possible_R_range[i]
            coeffs = [
            13.74,  # x^5
            -39.452, # x^4
            42.937,  # x^3
            -21.52,  # x^2
            5.2312 - (R / (R + 1)),  # x^1 coefficient adjusted for RSOL
            0.0831 - (d / (R + 1))   # Constant term adjusted for RSOL
            ]
            xs = [max(real_roots),d]
            ys = [rsol(max(real_roots),R,d), d]
            roots = np.roots(coeffs)
            real_roots = [r.real for r in roots if np.isreal(r) and 0 <= r.real <= 1]
            label = f"R = {R} and n = {len(real_roots)}"
            prev_R = R
            i += 1
        xs = [max(real_roots),d]
        ys = [rsol(max(real_roots),R,d), d]
        R_min = prev_R
        plt.plot(xs, ys, label=f"R_min RSOL (R_min = {prev_R:.5f}))")
    return R_min
        # print(f", R_min = {R_min:5f}", end="") 
    
    

def plot_mccabe_thiele(j, q, f, R, d, b):

    # Linear space for el and y=x
    x = np.linspace(0,1.0, 100)
    plt.figure(figsize=(12,12))
    
    #Plot y=x and EL from experimental data
    plt.plot(x,y_equals_x(x), label="y=x", color="black")
    plt.plot(x,el(x), label= "EL", color = "black")
    
    # Plot the feed, distillate, and bottoms mole fraction on y=x 
    plt.scatter(f, f, color="black", marker="x", s=50, label="_nolegend_")  # Red "X"
    plt.text(f, f, f"      Feed x = {f:.3}", fontsize=8, verticalalignment="center")  # Label the point
    
    plt.scatter(d, d, color="black", marker="x", s=50, label="_nolegend_")  # Red "X"
    plt.text(d, d, f"      Distillate x = {d:.3}", fontsize=8, verticalalignment="center")  # Label the point
    
    plt.scatter(b, b, color="black", marker="x", s=50, label="_nolegend_")  # Red "X"
    plt.text(b, b, f"      Bottoms x = {b:.3}", fontsize=8, verticalalignment="center")  # Label the point
    
    # Plot the feed line
    feed_line_start = f
    feed_line_end = fsolve(intersection_fl_el, f, args=(q, f))
    feed_line_end = feed_line_end[0]
    feed_line_range=np.linspace(feed_line_start, feed_line_end, 100)
    if not total[j]:
        plt.plot(feed_line_range,feed_line(feed_line_range,q,f), label="Feed line", color="brown")
    
    # Plot the rsol
    rsol_end = d
    rsol_start = fsolve(intersection_fl_rsol, f, args=(q, f,R,d))
    rsol_range = np.linspace(rsol_start, rsol_end, 100)
    plt.plot(rsol_range, rsol(rsol_range, R, d), label = "RSOL", color = "red")
    
    # Plot the ssol
    ssol_end = fsolve(intersection_rsol_fl, f, args=(R, d, q, f))
    ssol_end = ssol_end[0]
    ssol_start = b
    ssol_range = np.linspace(ssol_start, ssol_end, 100)
    plt.plot(ssol_range, ssol(ssol_range, ssol_start, ssol_start, ssol_end, feed_line(ssol_end, q, f)), label = "SSOL", color = "blue")

    # Iterate through McCabe Thiele
    stageCount = 0.0
    currentX = d
    currentY = y_equals_x(currentX)
    rsol_ssol_intersection = rsol_start
    
    # Calculate minimum reflux
    R_min = calculate_minimum_reflux(plt, R,d, feed_line_end,q,f)
    # print("R min is " + str(R_min))
    
    # While at position greater than bottoms mole fraction
    while (currentX > b):
        # Increase the stage count and record the previous fractions
        stageCount += 1 
        prevX = currentX
        prevY = currentY
        currentX = getELx(currentY)
        
        # Graph the current horizontal line of the method
        domain = np.linspace(currentX, prevX, 2)
        range = np.linspace(currentY, currentY, 2)
        plt.plot(domain, range, color="green")
        
        # If above SSOL RSOL intersection use the RSOL to iterate, else use the SSOL
        if (currentX > rsol_ssol_intersection):
            currentY = rsol(currentX,R,d)
        else:
            currentY = ssol(currentX,ssol_start, ssol_start, ssol_end, feed_line(ssol_end, q, f))
          
        # Graph the current vertical line of the method  
        domain = np.linspace(currentX, currentX, 2)
        y_to_plot = currentY
        if (currentY < y_equals_x(currentX)):
            y_to_plot = y_equals_x(currentX)
        
        range = np.linspace(y_to_plot, prevY, 2)
        plt.plot(domain, range,color="green")
        
    # Decrease by one since stage count over counts and then round up if req    
    stageCount -= 1
    frac = (prevX - b)/(prevX - currentX)
    if ((stageCount+frac) % 1 != 0):
        finalStageCount = stageCount + 1
    else:
        finalStageCount = stageCount + frac

    plt.grid()
    plt.legend()
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlim(0,1)
    plt.ylim(0,1)
    if (reflux_ratios[j] > 100):
        reflux_string = "Total"
    else:
        reflux_string = str(reflux_ratios[j])
        
    print(f", N = {finalStageCount - 1} tray + R, R_min = {R_min:5f}, R/R_min = {(R/R_min):.5f}")
    plt.title(f"{run_names[j]}, Reflux Ratio = {reflux_string}, q = {q:3f} "+
              f"N = {finalStageCount - 1} + R (rounded from {(stageCount + frac[0]):.3})")
    
    plt.xlabel("Liquid Mole Fraction Ethanol (x)")
    plt.ylabel("Vapor Mole Fraction Ethanol (y)")
    plt.savefig(os.path.join("./", f"{run_names[j]}.png"), dpi=300, bbox_inches="tight")
    

def main():
    results = dict()
    for i in range(len(feed_mole_fracs)):
        print(feed_mole_fracs[i])
        dp = dew_point_curve(feed_mole_fracs[i])
        bp = bubble_point_curve(feed_mole_fracs[i])
        q = calc_q(dp, bp, feed_temps[i])
        results[run_names[i]] = [dp, bp, q]
    
    i = 0
    for key,val in results.items():
        print(f"{key} -> Tdp = {val[0]:.5f} C, Tbp = {val[1]:.5f} C, q = {val[2]:.5f}", end = "")
        
        plot_mccabe_thiele(i, val[2], feed_mole_fracs[i], reflux_ratios[i], distillate_mole_fracs[i], bottoms_mole_fracs[i])
        i += 1
        
main()

