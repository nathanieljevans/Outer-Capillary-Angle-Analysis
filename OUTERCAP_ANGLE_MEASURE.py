# -*- coding: utf-8 -*-
"""
This scripts is intended to be used to locate capillary and chamber bores in a sequence of images then 
combine those values for use in finding outer capillary angle and position in reference to rotation axis 
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import skimage.transform as trans
from scipy import misc
from matplotlib.patches import Circle
from os import listdir
import os
from time import time

path = "C:\\Users\\evans\\Documents\\OUTER_CAP_MEASUREMENT_PICS\\C017-3945_001"
path2 = "C:\\Users\\evans\\Documents\\OUTER_CAP_MEASUREMENT_PICS\\C017-3945_002"

THRESHOLD = 0.25 # For mask segmentation
LEDGE_SEPARATION_DISTANCE = 0.577 # in  -- use calibrated tool during data acq. procedure 
UPPER_BORE_DIAM = 0.092 # in  -- original estimates, use only if no user input
LOWER_BORE_DIAM = 0.057 # in  -- original estimates, use only if no user input

''' 
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
            Delta_cap_xy, bore_diam, rcB, residu = create_composite_image(example_path+'\\'+opt, axarr[ii])
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
        
def create_composite_image(ledge_path, ax1): 
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
            cap, bore = get_circle_locations(misc.imread(ledge_path+'\\'+ img_path), img_path, outputs)
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
       
def cutoff(x): 
    shp = x.shape
    #THRESHOLD = np.mean(x) - 2*np.std(x)
#    print('shape ' + str(shp))
#    print('max ' + str(np.amax(x)))
#    print('min ' + str(np.amin(x)))
#    print('mean ' + str(np.mean(x)))
#    print('std dev ' + str(np.std(x)))
#    print('THRESHOLD: ' + str(THRESHOLD))
    c = np.zeros(shp)
    for i in range(shp[0]): 
        for j in range(shp[1]):
            if (x[i,j] > THRESHOLD):
                c[i,j] = 1
    return c
    
    
def get_circle_locations(image, name, outputs):
    try:
        summed = np.add(image[:,:,0], image[:,:,1])
        summed = np.add(summed, image[:,:,2])

        smoothed = filters.gaussian(summed, 5)       
        
        mask = cutoff(smoothed)
       
        edges = filters.sobel(mask)
        
        # Detect two radii
        hough_radii = np.arange(35, 75, 2)
        hough_res = trans.hough_circle(edges, hough_radii)
        
        # Select the most prominent 5 circles
        accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=1)
        cap_x = cx[0]
        cap_y = cy[0]
        cap_r = radii[0]
        
        # Detect two radii
        hough_radii = np.arange(200, 450, 2)
        hough_res = trans.hough_circle(edges, hough_radii)
        
        # Select the most prominent 5 circles
        accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=1)
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

from scipy import optimize
from matplotlib import cm, colors

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

def plot_data_circle(x,y, xc, yc, R):
    f = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    theta_fit = np.linspace(-pi, pi, 180)

    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')   
    # plot data
    plt.plot(x, y, 'r-.', label='data', mew=1)

    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.title('Least Squares Circle')
    
""" 
-----------------------------------------------------------------------------
""" 

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
