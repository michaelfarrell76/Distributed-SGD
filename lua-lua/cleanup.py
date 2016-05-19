#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Copy files from servers
"""

import sys
import os
import time


def child(ip_addr):
    if not os.path.exists('outputs/' + ip_addr):
            os.makedirs('outputs/' + ip_addr)
    os.system('(echo " echo starting; pkill torch; pkill lua; cd Distributed-SGD/lua-lua/; git pull; cd End-To-End-Generative-Dialogue/; git pull origin master; exit") | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/dist-sgd-sshkey michaelfarrell@%s' % ip_addr)
    os._exit(0)  


def main(arguments):
    with open('../client_list.txt') as f:
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        pids = []
        for line in f:
            # os.system('echo ' + line)
            newpid = os.fork()
            pids.append(newpid)
            if newpid == 0:
                if line[-1] ==  '\n':
                    child(line[:-1])
                else:
                    child(line)

    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))