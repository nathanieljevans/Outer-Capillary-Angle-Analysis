# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:59:38 2018

handle user input and data output

@author: evans
"""

from time import time

def main(): 
    tic = time()
    
    print('Example directories should be named in form: XXXX-YYY-AAAA-BBBB
    print('where X=OCA ID, Y=carrier_ID, A=upper_bore_diam (in), B=lower_bore_diam)
    print('example: 3963-B05-0.096-0.053')
    main_dir = input('Enter the directory that contains the example directories to be analyzed: ')
       
    
                
    for dir_ in filter(lambda x: x[-4] is not '.', os.listdir(main_dir)):
        print('Beginning dataset: ' + str(dir_))
        parse_dir_names(dir_)
        main(main_dir + '\\' + dir_)
    
    print('complete, time elapsed: ' + str(time() - tic)) 
    
def parse_dir_name(name): 
    vals = name.split('-')
    OCA = vals[0]
    car = vals[1]
    up_diam = val[2]
    low_diam = val[3]
    return OCA, car, up_diam, low_diam
    
    
if __name__ == '__main__' :
    main() 