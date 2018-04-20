# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:59:38 2018

handle user input and data output

@author: evans
"""

from time import time
import capillary_analysis as ca
import os 
import csv

def main(): 
    tic = time()
    
    print('Example directories should be named in form: XXX-YYYY-AAAA-BBBB-001')
    print('where X=carrier ID, Y=OCA_ID, A=upper_bore_diam (in), B=lower_bore_diam, 001-batch_iteration')
    print('example: B05-3963-0.096-0.053-001')
    main_dir = input('Enter the directory that contains the example directories to be analyzed: ')

    f = open(main_dir + '\\all_outputs.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(ca.OC_example.get_output_header())
    
    for dir_ in filter(lambda x: x[-4] is not '.', os.listdir(main_dir)):
        print('Beginning dataset: ' + str(dir_))
        OCA, CAR, UB_D, LB_D = parse_dir_name(dir_)
        example = ca.OC_example(main_dir + '\\' + dir_, OCA, CAR, upper_bore_ID=UB_D, lower_bore_ID=LB_D)
        example.load_images()
        example.analyze_images(method='lowexp')
        example.calculate_angle()
        example.calculate_offset()
        example.generate_and_save_plots()
        writer.writerow(example.print_output_line())
    f.close()
        
    
    print('complete, time elapsed: ' + str(time() - tic)) 
    
def parse_dir_name(name): 
    try: 
        vals = name.split('-')
        car = vals[0]
        OCA = vals[1]
        up_diam = vals[2]
        low_diam = vals[3]
        return OCA, car, up_diam, low_diam
    except: 
        raise TypeError('Check directory naming nomenclature')
        raise
    
    
if __name__ == '__main__' :
    main() 