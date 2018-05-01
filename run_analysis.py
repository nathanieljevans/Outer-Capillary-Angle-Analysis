# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:59:38 2018

handle user input and data output combination

@author: evans
"""

from time import time
import capillary_analysis as ca
import os 
import csv
from datetime import date

def main(): 
    tic = time()
    
    print('Example directories should be named in form: XXX-YYYY-AAAA-BBBB-001')
    print('where X=carrier ID, Y=OCA_ID, A=upper_bore_diam (in), B=lower_bore_diam, 001-batch_iteration')
    print('example: B05-3963-0.096-0.053-001')
    main_dir = input('Enter the directory that contains the example directories to be analyzed: ')
    method = input("Image analysis method to use: [both, otsu, lowexp] ")
    
    f = open(main_dir + '\\all_outputs-' + str(date.today()) + '.csv', 'w')
    writer = csv.writer(f)
    writer.writerow( ['dir name'] + ca.OC_example.get_output_header() ) # .insert(0,"dir name") 
    
    for dir_ in filter(lambda x: x[-4] is not '.', os.listdir(main_dir)):
        try: 
            print('Beginning dataset: ' + str(dir_))
            OCA, CAR, UB_D, LB_D = parse_dir_name(dir_)
            print('parsing vals: ')
            print('OCA: ' + str(OCA))
            print('CAR: ' + str(CAR))
            print('Upper bore diam: ' + str(UB_D))
            print('Lower bore diam: ' + str(LB_D))
            
            if (method == 'both' or method == 'lowexp'):
                example = ca.OC_example(main_dir + '\\' + dir_, OCA, CAR, upper_bore_ID=UB_D, lower_bore_ID=LB_D)
                example.load_images()
                example.analyze_images(method='lowexp')
                example.calculate_angle()
                example.calculate_offset()
                example.generate_and_save_plots()
                writer.writerow( [dir_] + example.print_output_line() )
            
            if (method == 'both' or method == 'otsu'):
                example_otsu = ca.OC_example(main_dir + '\\' + dir_, OCA, CAR, upper_bore_ID=UB_D, lower_bore_ID=LB_D)
                example_otsu.load_images()
                example_otsu.analyze_images(method='otsu')
                example_otsu.calculate_angle()
                example_otsu.calculate_offset()
                example_otsu.generate_and_save_plots()
                writer.writerow( [dir_] + example_otsu.print_output_line() )
        except: 
            print('Failed dir: ' + str(dir_))
            raise
    f.close()
        
    
    print('complete, time elapsed: ' + str(time() - tic)) 
    
def parse_dir_name(name): 
    try: 
        vals = name.split('-')
        car = vals[0]
        OCA = vals[1]
        up_diam = vals[2]
        low_diam = vals[3]
        return OCA, car, float(up_diam), float(low_diam)
    except: 
        raise TypeError('Check directory naming nomenclature')
        raise
    
    
if __name__ == '__main__' :
    main() 