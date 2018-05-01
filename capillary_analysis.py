# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:10:03 2018

@author: evans
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters,color
import skimage.transform as trans
from scipy import misc,optimize
from matplotlib.patches import Circle
from os import listdir
import os
from time import time
from math import factorial
from datetime import date


LEDGE_SEPARATION_DISTANCE = 0.577 # in  -- use calibrated tool during data acq. procedure 
UPPER_BORE_DIAM = 0.092 # in  -- original estimates, use only if no user input
LOWER_BORE_DIAM = 0.057 # in  -- original estimates, use only if no user input

# this still needs to get implemented. 
CAPILLARY_DIAM_UP_BOUND = 60
CAPILLARY_DIAM_LOW_BOUND = 30

# (threshold, radius lower bound, radius upper bound)
LOWER_BORE_SPECS = (0.3, 120, 210)
UPPER_BORE_SPECS = (0.3, 250, 400)

ROT_PATH_STD_DEV_CUTTOFF = 4
CAPILLARY_DIAM = 0.011811 # inches 

class OC_example(object):
    def __init__(self, example_path, OCA_ID, carrier_ID, upper_bore_ID=UPPER_BORE_DIAM, lower_bore_ID=LOWER_BORE_DIAM, ledge_separation=LEDGE_SEPARATION_DISTANCE):
        self.OCA = OCA_ID
        self.carrier = carrier_ID 
        self.upper_bore_ID = upper_bore_ID
        self.lower_bore_ID = lower_bore_ID
        self.ledge_separation = ledge_separation
        self.path = example_path
        self.date = date.today()
        self.fig, self.axarr = plt.subplots(ncols=2, nrows=5, figsize=(40, 40), sharex='all', sharey='all')
        self.sort_axarr()

        print("OCA ID: " + str(OCA_ID) + "  -  Carrier_ID: " + str(carrier_ID))
        
    def sort_axarr(self): 
        axs = []
        for ax in self.axarr.flatten(): 
            axs.append(ax)
        self.axarr = axs    
        
    def print_output_line(self): 
        line = [self.carrier,
                self.OCA, 
                self.angle_x,
                self.angle_y, 
                self.inlet_offset, 
                self.analyis_method,
                self.upper_ledge.cap_dist_from_rot_axis, 
                self.upper_ledge.avg_bore_diameter,
                self.upper_ledge.bore_diam_std,
                self.upper_ledge.cap_diam_std,
                self.upper_ledge.residuB,
                self.bottom_ledge.cap_dist_from_rot_axis, 
                self.bottom_ledge.avg_bore_diameter,
                self.bottom_ledge.bore_diam_std,
                self.bottom_ledge.cap_diam_std,    
                self.bottom_ledge.residuB] 
        return line 
        
    def get_output_header(): 
        header = ['carrier id',
                  'oca id', 
                  'angle x',
                  'angle y', 
                  'OC inlet - rot axis offset',
                  'image analysis method',
                  'upper bore cap dist from rot axis',
                  'upper bore avg bore diam',
                  'upper bore bore diam std',
                  'upper bore cap diam std',
                  'upper bore bore residue',
                  'lower bore cap dist from rot axis',
                  'lower bore avg bore diam',
                  'lower bore bore diam std',
                  'lower bore cap diam std',
                  'lower bore bore residue']
        return header
        
        
        
    def load_images(self): 
         DIRS = os.listdir(self.path)
         ii=0
         for opt in DIRS:     
            if (opt[-4] is not '.'):
                if (ii == 0): 
                    print("\tBOTTOM? " + str(opt))
                    self.bottom_ledge = ledge_set(self.path + '\\' + opt, self.axarr[0:5], LOWER_BORE_SPECS)
                    
                elif (ii == 1):
                    print("\tTOP? " + str(opt))
                    self.upper_ledge = ledge_set(self.path + '\\' + opt, self.axarr[5:], UPPER_BORE_SPECS)
                    
                else: 
                    raise TypeError("Too many folders in the ledge directory; There should only be a top and bottom folder (total 2)")
                ii+=1
                
    def calculate_angle(self): 
        lower = {"DX" : self.bottom_ledge.cap_dist_from_rot_axis[0], 
                 "DY" : self.bottom_ledge.cap_dist_from_rot_axis[1], 
                "bore diam" : self.bottom_ledge.avg_bore_diameter, 
                "rotation axis residue" : self.bottom_ledge.residuB, 
                "rotation path radius" : self.bottom_ledge.rcB, 
                "descrip" : "lower ledge values"}
        
        upper = {"DX" : self.upper_ledge.cap_dist_from_rot_axis[0], 
                 "DY" : self.upper_ledge.cap_dist_from_rot_axis[1], 
                "bore diam" : self.upper_ledge.avg_bore_diameter, 
                "rotation axis residue" : self.upper_ledge.residuB, 
                "rotation path radius" : self.upper_ledge.rcB, 
                "descrip" : "upper ledge values"}
        
        self.vals = {"upper ledge" : upper, "lower ledge" : lower, "Description " : "DX, DY, bore diam"}
        
        # get conversion from pixels to in
        print("upper ledge pix map using bore " + str( self.upper_bore_ID / self.vals['upper ledge']['bore diam']))
        print("lower ledge pix map using bore " + str( self.lower_bore_ID / self.vals['lower ledge']['bore diam']))
        self.vals["upper ledge"]['px map'] =  CAPILLARY_DIAM / self.upper_ledge.cap_diam
        self.vals['lower ledge']['px map'] =  CAPILLARY_DIAM / self.bottom_ledge.cap_diam
#        self.vals["upper ledge"]['px map'] =  ( self.upper_bore_ID / self.vals['upper ledge']['bore diam'])
#        self.vals['lower ledge']['px map'] =  ( self.lower_bore_ID / self.vals['lower ledge']['bore diam'])
        print('upper ledge pix map using cap: ' + str(CAPILLARY_DIAM / self.upper_ledge.cap_diam))
        print('lower ledge pix map using cap: ' + str(CAPILLARY_DIAM / self.bottom_ledge.cap_diam))
    
        self.vals['upper ledge']['DX-in'] = self.vals["upper ledge"]['px map']* self.vals['upper ledge']['DX']
        self.vals['upper ledge']['DY-in'] = self.vals["upper ledge"]['px map']* self.vals['upper ledge']['DY']
    
        self.vals['lower ledge']['DX-in'] = self.vals["lower ledge"]['px map']* self.vals['lower ledge']['DX']
        self.vals['lower ledge']['DY-in'] = self.vals["lower ledge"]['px map']* self.vals['lower ledge']['DY']

        self.vals['angle_x'] = np.degrees(np.arctan( ( self.vals['upper ledge']['DX-in'] - self.vals['lower ledge']['DX-in'] ) / LEDGE_SEPARATION_DISTANCE)) 
        self.vals['angle_y'] = np.degrees(np.arctan( ( self.vals['upper ledge']['DY-in'] - self.vals['lower ledge']['DY-in'] ) / LEDGE_SEPARATION_DISTANCE)) 
        self.angle_x = self.vals['angle_x']
        self.angle_y = self.vals['angle_y']
    
        self.vals['upper ledge']['mapped rotation path radius'] = self.vals['upper ledge']['rotation path radius']* self.vals['upper ledge']['px map']
        self.vals['lower ledge']['mapped rotation path radius'] = self.vals['lower ledge']['rotation path radius']* self.vals['lower ledge']['px map']
    
        f = open(self.path+'\\angle_calc-' + (self.path.split('\\')[-1]).replace('.','').replace(' ','') +'.txt', 'w')

        try:
            f.write(dict_printer(self.vals, 0))
        except:
            raise TypeError("Dictionary failed to print to file at example: " + str(self.path))
            raise
            
        f.close()
        return (self.angle_x, self.angle_y)
                    
    def calculate_offset(self):
        '''
   y1=1.5in   _ x1      
        /|   | \         
         |   |__\ x2   y2=0.933in
         |   |   \       /| 
         |/  |____\       |/
              inlet_offset
        '''
        f = open(self.path+'\\outputs-' + self.analyis_method + '-' + str(self.date) + '-' + (self.path.split('\\')[-1]).replace('.','').replace(' ','') +'.txt', 'a')
        x1 = self.vals['upper ledge']['DX-in']
        y1 = 1.5 # in - distance from lead in to upper ledge
        x2 = self.vals['lower ledge']['DX-in']
        y2 = y1 - LEDGE_SEPARATION_DISTANCE # distance from lead in to lower ledge     
        m = (y2-y1) / (x2 - x1)
        b = y1 - m * x1 
        self.inlet_offset = -b/m
        
        f.write('inlet offset: ' + str(self.inlet_offset))
        f.close()
        
        
    def generate_and_save_plots(self): 
        self.fig.suptitle('OCA: ' + self.OCA + ' - Carrier: ' + self.carrier + 
                          '\nDate: ' + str(self.date) + 
                          '\nX Angle: ' + str(self.angle_x) + 
                          '\nRot. Axis - Cap. Inlet X offset: ' + str(self.inlet_offset), fontsize=12)
        
        self.fig.savefig(self.path + '\\outputs-' + self.analyis_method + '-' + str(self.date) + '.png')
        
        
    def analyze_images(self, method='otsu'): 
        self.analyis_method = method
        self.bottom_ledge.analyze(method=method)
        self.upper_ledge.analyze(method=method)
        self.bottom_ledge.calculate_ledge_features()
        self.upper_ledge.calculate_ledge_features()
                    
                    


class ledge_set(object): 
    def __init__(self, path, axes, specs):
        self.name = path.split('\\')[-1]    # upper/lower
        self.path = path
        self.imgs = []
        
        self.axes = axes[0]
        self.img_axes = axes[1:]
        self.specs = specs
        
        self.cap_xs = []
        self.cap_ys = [] 
        self.cap_rads = []

        self.bore_xs = []
        self.bore_ys = []
        self.bore_rads = []
        
        self.load_imgs()
        
    def load_imgs(self):
        img_names = os.listdir(self.path)
        i = 0
        for name in img_names: 
            if (name[-1]=='p'):            
                self.imgs.append( image(self.path + '\\' + name, self.img_axes[i], self.specs) )
                i+=1 
        if (i != 4): 
            raise TypeError("Too few or too many images in ledge set folder: " + str(self.name))
    
    def analyze(self, method='otsu'): 
        if (method == 'otsu'): 
            for img in self.imgs: 
                img.analyze_otsu()
        elif(method == 'lowexp'): 
            for img in self.imgs: 
                img.analyze_lowexp()
        else: 
            raise TypeError('Must use analysis method either: \'otsu\' or \'lowexp\'')
            
    def calculate_ledge_features(self):
        
        # capillary analysis-XXX() needs to have been run before this method can function
        try:
            for img in self.imgs: 
                self.cap_xs.append(img.cap_x)
                self.cap_ys.append(img.cap_y)
                self.cap_rads.append(img.cap_r)
                
                self.bore_xs.append(img.bore_x)
                self.bore_ys.append(img.bore_y)
                self.bore_rads.append(img.bore_r)
        except: 
            raise ValueError('An image analysis method needs to be run on the image before calculating ledge features')
            raise
        
        # add overlaid (non-aligned) circles to axes in red
        self.axes.imshow(self.imgs[0].img)
        self.axes.set_title('composite img: ' + self.name)
        for cx,cy,cr,bx,by,br in zip(self.cap_xs, self.cap_ys, self.cap_rads, self.bore_xs, self.bore_ys, self.bore_rads):
            self.axes.add_patch(Circle((cx,cy),cr, color='r', fill=False))
            self.axes.add_patch(Circle((bx,by),br, color='r', fill=False))
            
        # we want to align the images to the capillary. Capillary is our reference feature as it does not rotate with the carrier. This accounts for carrier movement between pictures
        # use first image cap x,y as ref 
        print("\tAligning capillaries...")
        x_ref = self.cap_xs[0]
        y_ref = self.cap_ys[0]
        for i,x,y in zip(range(1,4),self.cap_xs[1:],self.cap_ys[1:]): 
            dx = x-x_ref
            dy = y-y_ref
            self.cap_xs[i] = x-dx 
            self.cap_ys[i] = y-dy
            self.bore_xs[i] = self.bore_xs[i]-dx
            self.bore_ys[i] = self.bore_ys[i]-dy
                
         # see how well fit the circle is... ie was it rotated concentricly or was it bumped/shifted during rot
         # if the rot points are too tightly clustered (ie extremely concentric rotation) then the best fit circle plots it as a line (massive circle)
         # to fix it, check to see if its tightly clustered, if it is, don't do a circle fit
        try: 
            print('\tCalculating rotation axis...')
            self.xcC,self.ycC,self.rC,self.residuC = leastsq_circle(self.cap_xs, self.cap_ys)
            if (check_rot_points_tightness(self.bore_xs, self.bore_ys)): 
                self.xcB = np.average(self.bore_xs)
                self.ycB = np.average(self.bore_ys)
                self.rcB = ( np.std(self.bore_xs) + np.std(self.bore_ys) ) / 2 
                self.residuB = -1
            else: 
                self.xcB, self.ycB, self.rcB, self.residuB = leastsq_circle(self.bore_xs, self.bore_ys) 
        except: 
            raise ValueError('Failed trying to calculate the least sq best fit circle for cap and bore rot.')
            raise
                
        #rotation_axis_xy = ((max(bore_xs) + min(bore_xs)) / 2, (max(bore_ys) + min(bore_ys)) / 2)
        rotation_axis_xy = (self.xcB,self.ycB)
        #avg_cap_xy = ( np.average(self.cap_xs), np.average(self.cap_ys) )
        # use cap xy from first photo, the value that all the caps are aligned too.
        self.cap_dist_from_rot_axis = ( (rotation_axis_xy[0] - x_ref), (rotation_axis_xy[1] - y_ref) ) 
        self.avg_bore_diameter = 2* (sum(self.bore_rads) / len(self.bore_rads))
        self.bore_diam_std = np.std(self.bore_rads)
        self.cap_diam_std = np.std(self.cap_rads)
        self.cap_diam = 2*np.average(self.cap_rads)
            
        for cx,cy,cr,bx,by,br in zip(self.cap_xs, self.cap_ys, self.cap_rads, self.bore_xs, self.bore_ys, self.bore_rads):
            #add aligned capillary and bore positions in green
            self.axes.add_patch(Circle((cx,cy),cr, color='g', fill=False))
            self.axes.add_patch(Circle((bx,by),br, color='g', fill=False))
            #add bore xc xy as yellow 
            self.axes.add_patch(Circle((bx,by),5, color='y', fill=True))
            #add bore rotation path as yellow 
            self.axes.add_patch(Circle((self.xcB,self.ycB),self.rcB,color='y',fill=False))
        
        # add rotation center in white
        self.axes.add_patch(Circle(rotation_axis_xy, 5, color='w', fill=True))


class image(object): 
    def __init__(self, path, ax, specs): 
        self.path = path
        self.name = path.split('\\')[-1]
        self.img = plt.imread(path)
        self.axes = ax
        self.specs = specs
    
    # works well for variable light conditions and low contrast
    def analyze_otsu(self): 
        try:        
            # convert to grey 
            grey = color.rgb2gray(self.img)
            
            #smooth to remove noise, high noise makes hough transform take significantly more time
            grey_smooth = filters.gaussian(grey,3)
            
            # threshold via otsu histogram binning
            otsu = grey_smooth > filters.threshold_otsu(grey_smooth)
            
            # calculate edges via sobel method 
            edges = filters.sobel(otsu)
            
            print('\tcalculating hough transform...')
            
            # Detect two radii
            hough_radii = np.arange(CAPILLARY_DIAM_LOW_BOUND, CAPILLARY_DIAM_UP_BOUND, 2)
            hough_res = trans.hough_circle(edges, hough_radii)
            
            # Select the most prominent circle
            accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii,
                                                       total_num_peaks=1)
            self.cap_x = cx[0]
            self.cap_y = cy[0]
            self.cap_r = radii[0]
            
            # Detect two radii
            hough_radii = np.arange(self.specs[1], self.specs[2], 2)
            hough_res = trans.hough_circle(edges, hough_radii)
            
            # Select the most prominent circle
            accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii,
                                                       total_num_peaks=1)
            print("\taccums, cx, cy, radii " + str( (accums, cx, cy, radii) ))
            self.bore_x = cx[0]
            self.bore_y = cy[0]
            self.bore_r = radii[0]
        
            self.axes.imshow(self.img)
            self.axes.set_title(str(self.name))
            self.axes.add_patch(Circle((self.bore_x,self.bore_y), self.bore_r, color='r', fill=False))
            self.axes.add_patch(Circle((self.cap_x,self.cap_y), self.cap_r, color='r', fill=False))
            
        except: 
            raise ValueError("Failed during Otsu analysis of img: " + str(self.name))
            raise 
            
    # basic thresholding that works well for low exposure, high contrast photos
    # only run one analysis method per image object, running a second will overwrite 
    def analyze_lowexp(self): 
        try:
            summed = np.add(np.add(self.img[:,:,0], self.img[:,:,1]), self.img[:,:,2]) 
        
            smoothed = filters.gaussian(summed, 5)       
            
            mask = cutoff(smoothed, self.specs)
            
            edges = filters.sobel(mask)
            
            # Detect two radii
            hough_radii = np.arange(CAPILLARY_DIAM_LOW_BOUND, CAPILLARY_DIAM_UP_BOUND, 2)
            hough_res = trans.hough_circle(edges, hough_radii)
            
            print('\tcalculating hough transform...')
            # Select the most prominent 1 circles
            accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
            self.cap_x = cx[0]
            self.cap_y = cy[0]
            self.cap_r = radii[0]
            
            # Detect two radii
            hough_radii = np.arange(self.specs[1], self.specs[2], 2)
            hough_res = trans.hough_circle(edges, hough_radii)
            
            # Select the most prominent 5 circles
            accums, cx, cy, radii = trans.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
            
            print("\tbore - accums, cx, cy, radii " + str( (accums, cx, cy, radii) ))
            self.bore_x = cx[0]
            self.bore_y = cy[0]
            self.bore_r = radii[0]
        
            self.axes.imshow(self.img)
            self.axes.add_patch(Circle((self.bore_x, self.bore_y), self.bore_r, color='r', fill=False))
            self.axes.add_patch(Circle((self.cap_x, self.cap_y), self.cap_r, color='r', fill=False))
            
        except: 
            raise ValueError("Failed during lowexp analysis of img: " + str(self.name))
            raise 

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ''' 
        ---------------------------- additional methods -----------------------
        ''' 
        
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
        
        
def cutoff(x, specs): 
    shp = x.shape
    THRESHOLD = specs[0]
    c = np.zeros(shp)
    for i in range(shp[0]): 
        for j in range(shp[1]):
            if (x[i,j] > THRESHOLD):
                c[i,j] = 1
    return c
        
    
""" 
-----------------------------------------------------------------------------
Taken from: 
        http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
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
    print('xs std: ' + str(np.std(xs)))
    print('ys std: ' + str(np.std(ys)))
    if (np.std(xs) <= ROT_PATH_STD_DEV_CUTTOFF or np.std(ys) <= ROT_PATH_STD_DEV_CUTTOFF): 
        return True
    else: 
        return False 
        
        
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm     
    
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