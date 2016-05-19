#!/usr/bin/env python 
# -*- coding: utf-8 -*-


"""Copy files from servers
"""

from __future__ import print_function

import sys
import os
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
import matplotlib.pyplot as plt


class Print:
    def red(self, prt): print("\033[91m{}\033[00m" .format(prt), end="")
    def green(self, prt): print("\033[92m{}\033[00m" .format(prt), end="")
    def yellow(self, prt): print("\033[93m{}\033[00m" .format(prt), end="")
    def lightpurple(self, prt): print("\033[94m{}\033[00m" .format(prt), end="")
    def purple(self, prt): print("\033[95m{}\033[00m" .format(prt), end="")
    def cyan(self, prt): print("\033[96m{}\033[00m" .format(prt), end="")
    def lightgray(self, prt): print("\033[97m{}\033[00m" .format(prt), end="")
    def black(self, prt): print("\033[98m{}\033[00m" .format(prt), end="")

class Result:
    def __init__(self, floc):
        self.results = []
        self.floc = floc
        self.loc_split = floc.split('/')
        self.fname = self.loc_split[-1]
        self.ip_addr = self.loc_split[-2]
        self.no_ext = self.fname.split('.')[0]
        self.ada_grad, self.n_proc, self.loc =  self.no_ext.split('_')
        self.n_proc = int(self.n_proc)

        if self.ada_grad == 'ada':
            self.ada_grad = 'ada grad SGD'
        else:
            self.ada_grad = 'simple SGD'

        if self.loc == 'rem':
            self.loc = 'remotely'
        else:
            self.loc = 'locally'

        self.description = '%d processes, %s, running %s' % (self.n_proc, self.ada_grad, self.loc)

    def add_result(self, result):
        self.results.append(result)

    def get_data(self, max_epoch, min_t):
        return [result.time_ellapse for result in self.results if (max_epoch is None  or result.epoch <= max_epoch) and (min_t is None or result.time_ellapse >= min_t)], [np.log(result.perplexity) for result in self.results if (max_epoch is None  or result.epoch <= max_epoch) and (min_t is None or result.time_ellapse >= min_t)]
    
    def graph(self, close = True, out_name = None, max_epoch = None, min_t = None):
        times, log_perps = self.get_data(max_epoch, min_t)
        
        plt.ylabel('Log perplexity')
        plt.xlabel('Time (s)')

        plt.title(self.description)
        plt.plot(times, log_perps, label = self.description)
        
        if close:
            if out_name == None:
                out_name = "/".join(self.loc_split[:-1]) + '/' + self.no_ext + '.png'
        
            plt.savefig(out_name)
            plt.clf()
            plt.cla()
            plt.close()


    def display(self):
        Print().green('Results for file %s \n' % self.floc) 

        Print().lightpurple('Number of processes: ')
        print(self.n_proc)

        Print().lightpurple('SGD type: ')
        print(self.ada_grad)

        Print().lightpurple('Running location: ')
        print(self.loc)

        Print().lightpurple('Server: ')
        print(self.ip_addr)

        if len(self.results) == 0:
            Print().red('No results\n')
            return

        Print().lightpurple('Number of batches: ')
        print(self.results[0].n_batch)

        epoch = -1
        for result in self.results:
            if result.epoch != epoch:
                epoch = result.epoch
                Print().yellow('Epoch: %d\n' % epoch)
            result.display()

class DataPoint:
    def __init__(self, line):
        # Store the line itself
        self.line = line

        # The epoch we're on
        self.epoch = int(self.clean_match('Epoch: (.*?), Batch:', line))
        
        # Current batch, total number of batches, current batchsuze
        self.batch_str = self.clean_match('Batch: (.*?), Batch size:', line)
        batch_splt = str.split(self.batch_str, '/')
        self.batch, self.n_batch = [int(ind) for ind in batch_splt]
        self.batch_size = int(self.clean_match('Batch size: (.*?), LR:', line))

        self.learning_rate = float(self.clean_match('LR: (.*?), PPL: ', line))
        
        self.perplexity = float(self.clean_match('PPL: (.*?), |Param|:', line))
        
        self.speed = self.clean_match('Training: (.*?) total/source/target', line)

        self.time_ellapse = int(str.split(line)[-1])


    def clean_match(self, pattern, string):
        res = re.findall(pattern, string)
        return filter(lambda x: x != '', res)[0]
    def display(self):
        args = (self.batch, self.perplexity, self.time_ellapse)
        print('Batch: %d, perplexity: %.2f, time: %d\n' % args, end = "")

class Results:
    def __init__(self):
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    def graph(self, location = None, max_epoch = None, min_t = None):
        for result in self.results:
            if location == None or result.loc == location:
                result.graph(close = False, max_epoch = max_epoch, min_t = min_t)
        if location == None:
            out_name = "All.png"
        else:
            out_name = location + ".png"
        plt.title(location)
        plt.legend(bbox_to_anchor=(1.05, 1))

        plt.savefig(out_name)
        plt.clf()
        plt.cla()
        plt.close()



def process_file(path_to_file):
    result = Result(path_to_file)
    with open(path_to_file) as f:
        for line in f:
            if 'total/source/target' in line:                
                # Parse the line into a DataPoint object 
                data_point = DataPoint(line)

                # Add the datapoint to the result
                result.add_result(data_point)
                
        result.display()
        result.graph()
    return result
       
               

def main(arguments):

    while True:
        print('Copying over files')
        # Updating files
        os.system('python copy_files.py')

        import time
        time.sleep(3)

        # hold the results
        results = Results()

        # Get all folders of ip addresses
        for ip_fold in os.walk('outputs'):

            # Find the .txt files
            for file in os.listdir(ip_fold[0]):
                if file.endswith(".txt") and len(file.split('_')) == 3:

                    # Full path to the file
                    full_path = ip_fold[0] + '/' + file

                    result = process_file(full_path)

                    results.add_result(result)
        
        results.graph(location = 'locally', max_epoch = 7)
        results.graph(location = 'remotely', min_t = 50, max_epoch = 10)
        time.sleep(20)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))