# -*- coding: utf-8 -*-
"""
This scripts is intended to be used to locate capillary and chamber bores in a sequence of images then 
combine those values for use in finding outer capillary angle and position in reference to rotation axis 
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import skimage.transform as trans
from scipy import misc,optimize
from matplotlib.patches import Circle
from os import listdir
import os
from time import time
from math import factorial

LEDGE_SEPARATION_DISTANCE = 0.577 # in  -- use calibrated tool during data acq. procedure 
UPPER_BORE_DIAM = 0.092 # in  -- original estimates, use only if no user input
LOWER_BORE_DIAM = 0.057 # in  -- original estimates, use only if no user input

# this still needs to get implemented. 
CAPILLARY_DIAM_UP_BOUND = 60
CAPILLARY_DIAM_LOW_BOUND = 30

TOP_BORE_UP_BOUND = 400
TOP_BORE_LOW_BOUND = 250

LOWER_BORE_UP_BOUND = 210 
LOWER_BORE_LOW_BOUND = 120

TOP_BORE_THRESHOLD = 0.3 # was .25
LOWER_BORE_THRESHOLD = 0.3 #0.2 was over sat

''' 
0.20 thresh works well for bottom but 0.25 works better for top
This is some kind of documentation
'''
def main(example_path): 
    DIRS = os.listdir(example_path)
    deltas = []
    diams = []
    rcs = []
    residus = []

    ii = 0
    fig1, axarr = plt.subplots(ncols=1, nrows=2, figsize=(15, 15))
    for opt in DIRS:     
        if (opt[-4] is not '.'):
            if (ii == 0): 
                print("BOTTOM? " + str(opt))
                Delta_cap_xy, bore_diam, rcB, residu = create_composite_image(example_path+'\\'+opt, axarr[ii], (LOWER_BORE_THRESHOLD,LOWER_BORE_LOW_BOUND,LOWER_BORE_UP_BOUND) )
            else: 
                print("TOP? " + str(opt))
                Delta_cap_xy, bore_diam, rcB, residu = create_composite_image(example_path+'\\'+opt, axarr[ii], (TOP_BORE_THRESHOLD,TOP_BORE_LOW_BOUND,TOP_BORE_UP_BOUND) )
            rcs.append(rcB)
            residus.append(residu)
            deltas.append(Delta_cap_xy)
            diams.append(bore_diam)
            ii+=1
    fig1.savefig(example_path+'\\'+'composite.png')
    plt.close('all')

    upper = {"DX" : deltas[0][0], "DY" : deltas[0][1], "bore diam" : diams[0], "rotation axis residue" : residus[0], "rotation path radius" : rcs[0], "descrip" : "upper ledge values"}
    lower = {"DX" : deltas[1][0], "DY" : deltas[1][1], "bore diam" : diams[1], "rotation axis residue" : residus[1], "rotation path radius" : rcs[1], "descrip" : "lower ledge values"}

    if (diams[1] > diams[0]): 
        temp = lower
        lower['descrip'] = 'upper ledge values'
        upper['descrip'] = 'lower ledge values'
        temp = lower
        lower = upper 
        upper = temp
    
    vals = {"upper ledge" : upper, "lower ledge" : lower, "Description " : "DX, DY, bore diam"}
    
    # get conversion from pixels to in
    vals["upper ledge"]['px map'] = UPPER_BORE_DIAM / vals['upper ledge']['bore diam']
    vals['lower ledge']['px map'] = LOWER_BORE_DIAM / vals['lower ledge']['bore diam']

    vals['upper ledge']['DX-in'] = vals["upper ledge"]['px map']*vals['upper ledge']['DX']
    vals['upper ledge']['DY-in'] = vals["upper ledge"]['px map']*vals['upper ledge']['DY']

    vals['lower ledge']['DX-in'] = vals["lower ledge"]['px map']*vals['upper ledge']['DX']
    vals['lower ledge']['DY-in'] = vals["lower ledge"]['px map']*vals['upper ledge']['DY']

    vals['angle_x'] = np.degrees(np.arctan( ( vals['upper ledge']['DX-in'] - vals['lower ledge']['DX-in'] ) / LEDGE_SEPARATION_DISTANCE)) 
    vals['angle_y'] = np.degrees(np.arctan( ( vals['upper ledge']['DY-in'] - vals['lower ledge']['DY-in'] ) / LEDGE_SEPARATION_DISTANCE)) 

    vals['upper ledge']['mapped rotation path radius'] = vals['upper ledge']['rotation path radius']*vals['upper ledge']['px map']
    vals['lower ledge']['mapped rotation path radius'] = vals['lower ledge']['rotation path radius']*vals['lower ledge']['px map']

    f = open(example_path+'\\angle_calc-' + (example_path.split('\\')[-1]).replace('.','').replace(' ','') +'.txt', 'w')
    f.write(str(vals))
    try:
        f.write('\n\n\n\n\n')
        f.write(dict_printer(vals, 0))
    except:
        print('dict writer failed')
    f.close()
    
def dict_printer(dic, lvl): 
    s = '\n' + '\t'*lvl 
    if (type(dic) is dict):
        for i,key in enumerate(dic.keys()): 
            if (i > 0): 
                s += '\n' + '\t'*lvl
            s += key + ' { ' + dict_printer(dic[key], lvl+1)
    else: 
        s += '[ ' + str(dic) + ' ]'
    return s 
        
def create_composite_image(ledge_path, ax1, specs): 
    outputs = ledge_path+'\\outputs'
    if  not os.path.isdir(outputs):
        os.makedirs(outputs)
    
    f = open(outputs+'\\composite_outputs.txt','w')
    f.write('IMG NAME -- capillary: (x,y,radius)) -- bore: (x,y, radius)\n')
    img_paths = listdir(ledge_path)
    cap_loc = []
    bore_loc = []
    cap_xs = []
    cap_ys = []
    bore_xs = []
    bore_ys = [] 
    bore_rads = []
    for img_path in img_paths: 
        if (img_path[-1]=='p'):
            cap, bore = get_circle_locations(misc.imread(ledge_path+'\\'+ img_path), img_path, outputs, specs)
            cap_loc.append(cap)
            cap_xs.append(cap[0])
            cap_ys.append(cap[1])
            bore_loc.append(bore)
            bore_xs.append(bore[0])
            bore_ys.append(bore[1])
            bore_rads.append(bore[2])
    
            
    # we want to align the images to the capillary. Capillary should be stationary in these images. 
    # use first image cap x,y as ref 
    print("Aligning capillaries...")
    x_ref = cap_xs[0]
    y_ref = cap_ys[0]
    for i,x,y in zip(range(1,4),cap_xs[1:],cap_ys[1:]): 
        dx = x-x_ref
        dy = y-y_ref
        cap_xs[i] = x-dx 
        cap_ys[i] = y-dy
        bore_xs[i] = bore_xs[i]-dx
        bore_ys[i] = bore_ys[i]-dy
    
    adjusted_cap = [] 
    for x,y,r in zip(cap_xs,cap_ys,cap_loc): 
        adjusted_cap.append([x,y,r[2]])
        
    adjusted_bore = []
    for x,y,r in zip(bore_xs,bore_ys,bore_loc): 
        adjusted_bore.append([x,y,r[2]])
            
     # see how well fit the circle is... ie was it rotated concentricly or was it bumped/shifted during rot
    try: 
        xcC,ycC,rC,residuC = leastsq_circle(cap_xs, cap_ys)
        if (check_rot_points_tightness(bore_xs, bore_ys)): # need this to fix the bug of extremely small bearing rot leads to inaccurate circle fit 
            xcB = np.average(bore_xs)
            ycB = np.average(bore_ys)
            rcB = ( np.std(bore_xs) + np.std(bore_ys) ) / 2 
            residuB = -1
        else: 
            xcB, ycB, rcB, residuB = leastsq_circle(bore_xs, bore_ys) 
        print('cap : ' + str([xcC,ycC,rC,residuC]))
        print('bore : ' + str([xcB, ycB, rcB, residuB]))
    except: 
        print('failed')
            
    #rotation_axis_xy = ((max(bore_xs) + min(bore_xs)) / 2, (max(bore_ys) + min(bore_ys)) / 2)
    rotation_axis_xy = (xcB,ycB)
    avg_cap_xy = ( (sum(cap_xs)/len(cap_xs)),(sum(cap_ys)/len(cap_ys))  )
    cap_dist_from_rot_axis = ( abs(rotation_axis_xy[0] - avg_cap_xy[0]), abs(rotation_axis_xy[1] - avg_cap_xy[1]) ) 
    avg_bore_diameter = 2* (sum(bore_rads) / len(bore_rads))
    
    
    img = misc.imread(ledge_path+'\\'+img_paths[0])
    ax1.imshow(img)
    for cap, bore,img_name in zip(cap_loc,bore_loc,img_paths):
        # add the NON-adjusted cap and bore location (red)
        ax1.add_patch(Circle((bore[0],bore[1]),bore[2], color='r', fill=False))
        ax1.add_patch(Circle((cap[0],cap[1]),cap[2], color='r', fill=False))
        f.write(img_name + '-- capillary: ' + str(cap) + '-- bore: ' + str(bore) + '\n')
        
    for cap, bore,img_name in zip(adjusted_cap,adjusted_bore,img_paths):
        # add the adjusted cap and bore calculated locations (green)
        ax1.add_patch(Circle((bore[0],bore[1]),bore[2], color='g', fill=False))
        ax1.add_patch(Circle((cap[0],cap[1]),cap[2], color='g', fill=False))
        #add bore xc xy as yellow 
        ax1.add_patch(Circle((bore[0],bore[1]),5, color='y', fill=True))
        #add bore rotation path as yellow 
        ax1.add_patch(Circle((xcB,ycB),rcB,color='y',fill=False))

        f.write(img_name + '--adjusted capillary: ' + str(cap) + '--adjusted bore: ' + str(bore) + '\n')
        
    f.write('\n\nMean Capillary XY: ' + str(avg_cap_xy)+'\n')
    f.write('Rotation axis XY: ' + str(rotation_axis_xy))
    f.write("\nPixel Distance between cap. axis and rot. axis (DX, DY): " + str(cap_dist_from_rot_axis)+'\n')
    f.write("Average bore diameter: " + str(avg_bore_diameter))
    f.close()
    
    # add rotation center in white
    ax1.add_patch(Circle(rotation_axis_xy, 5, color='w', fill=True))
    return cap_dist_from_rot_axis, avg_bore_diameter, rcB, residuB
       
def cutoff(x, specs): 
    shp = x.shape
#    THRESHOLD = np.amin(x) + 0.05
#    print('shape ' + str(shp))
#    print('max ' + str(np.amax(x)))
#    print('min ' + str(np.amin(x)))
#    print('mean ' + str(np.mean(x)))
#    print('std dev ' + str(np.std(x)))
    #print('THRESHOLD: ' + str(THRESHOLD))
    
#    fx_unsmoothed = x.flatten()
    # plt.hist(x.ravel())
#    fx_unsmoothed.sort()
#    fx = savitzky_golay(fx_unsmoothed)
#    
#    d_fx = np.gradient(fx)
##    d2_fx = np.gradient(d_fx)
##    max_curve = np.amax(d2_fx[np.where(fx >= 0.3)[0][0] : np.where(fx >= 0.7)[0][0]])
##    a_val = fx[np.where(d2_fx == max_curve)[0][0]]
#
#    # want the first point after it's peak at which the deriv is X% of it's max val 
#    dfx_peak_i = np.where(d_fx == np.amax(d_fx))[0]
#    
#    print('dfx peak ' + str(dfx_peak_i))
#    eop_is = np.where(d_fx <= 0.1*np.amax(d_fx))
#    print(eop_is)
#    eop_i = -1
#    for i in eop_is[0]: 
#        if (i >= dfx_peak_i): 
#            eop_i = i
#            break
#        
#    fx_mid = (fx[int(eop_i)] - fx[0])/2 + fx[0]
#            
#    print('even out point index ' + str(eop_i))
#    eop_dfx = d_fx[eop_i]
#    eop_fx = fx[eop_i]
#    
#    print('even out point (dfx, fx): ' + str( (eop_dfx, eop_fx)))
##    print("a val: " + str(a_val))
#    THRESHOLD = fx_mid
#    plt.plot(fx,color='g' )
#    plt.plot(fx_unsmoothed,color='r')
##    plt.plot([a_val]*len(fx),color='y')
#    plt.plot([eop_fx]*len(fx),color='r')
#    plt.plot([fx_mid]*len(fx),color='g')
#    plt.show()
#    plt.plot(d_fx,color='b')
#    plt.plot([eop_dfx]*len(fx),color='r')
#    plt.show()

    THRESHOLD = specs[0]
    c = np.zeros(shp)
    for i in range(shp[0]): 
        for j in range(shp[1]):
            if (x[i,j] > THRESHOLD):
                c[i,j] = 1
    return c
        
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm      
    
def get_circle_locations(image, name, outputs, specs):
    try:
        summed = np.add(image[:,:,0], image[:,:,1])
        summed = np.add(summed, image[:,:,2])

        smoothed = filters.gaussian(summed, 5)       
        
        mask = cutoff(smoothed, specs)
#        plt.imshow(mask)
#        plt.show()
        
        edges = filters.sobel(mask)
        
        # Detect two radii
        hough_radii = np.arange(CAPILLARY_DIAM_LOW_BOUND, CAPILLARY_DIAM_UP_BOUND, 2)
        hough_res = trans.hough_circle(edges, hough_radii)
        
        # Select the most prominent 1 circles
        accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=1)
        cap_x = cx[0]
        cap_y = cy[0]
        cap_r = radii[0]
        
        # Detect two radii
        hough_radii = np.arange(specs[1], specs[2], 2)
        hough_res = trans.hough_circle(edges, hough_radii)
        
        # Select the most prominent 5 circles
        accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=1)
        print("accums, cx, cy, radii " + str( (accums, cx, cy, radii) ))
        bore_x = cx[0]
        bore_y = cy[0]
        bore_r = radii[0]
    
        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        ax1.imshow(image)
        ax1.add_patch(Circle((bore_x,bore_y),bore_r, color='r', fill=False))
        ax1.add_patch(Circle((cap_x,cap_y),cap_r, color='r', fill=False))
        fig1.savefig(outputs+'\\'+name[:-4]+'-circled.png')
        plt.close('all')
        return (cap_x, cap_y, cap_r), (bore_x,bore_y,bore_r)
    except: 
        print('failed at ' + str(name))
        print('expected outputs ' + str(outputs))
        print()
        raise
 
""" 
TAKEN FROM : https://gist.github.com/lorenzoriano/6799568 -------------------------
""" 

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu
    
""" 
-----------------------------------------------------------------------------
VVV taken from: http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
""" 

def savitzky_golay(y, window_size=100001, order=1, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    
    """ 
    ___________________________________________________
    """
def check_rot_points_tightness(xs, ys):
    print('xs std : ' + str(np.std(xs)))
    print('ys std : ' + str(np.std(ys)))
    if (np.std(xs) <= 2 and np.std(ys) <= 2): 
        return True
    else: 
        return False

#main('C:\\Users\\evans\\Documents\\OUTER_CAP_MEASUREMENT_PICS\\OCI-003\\C071-0000003972 (0.092-0.057)')

tic = time()

main_dir = input('Enter the directory that contains the test directories to be analyzed, or if this program is in said location, leave blank: ')
if (main_dir is None): 
    main_dir = './'            

upper_bore = input ('what is the upper bore diameter (inches) enter in form 0.XXX: ' )
lower_bore = input ('what is the lower bore diameter (inches) enter in form 0.XXX: ' )
if (upper_bore is not None): 
    UPPER_BORE_DIAM = float(upper_bore)
if (lower_bore is not None): 
    LOWER_BORE_DIAM = float(lower_bore)
            
for dir_ in filter(lambda x: x[-4] is not '.', os.listdir(main_dir)):
    print(dir_)
    main(main_dir + '\\' + dir_)

print('complete, time elapsed: ' + str(time() - tic))