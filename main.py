#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:57:34 2018

  * Generator of matrix elements in Antoine format

@author: miguel
"""

from sys import argv

from helpers.TBME_Runner import TBME_Runner
from helpers.TBME_SpeedRunner import TBME_SpeedRunner


if __name__ == "__main__":
    
    terminal_args_given = argv
    if len(terminal_args_given) > 1:
        print(" [2B_MatrixElements.main] Running SpeedRunner Suite, argv:", 
              terminal_args_given)
        assert terminal_args_given[1].endswith('.xml'), \
            f"Only XML input files accepted. Got [{terminal_args_given[1]}]"
        _runner = TBME_SpeedRunner(argv[1], verbose=False)
        _runner.run()
    else:    
        # _runner = TBME_SpeedRunner(filename='input_B1.xml')
        # _runner = TBME_Runner(filename='input.xml')
        # _runner = TBME_SpeedRunner(filename='input.xml')
        # _ = 0
        # _runner = TBME_Runner(filename='input_M3Y.xml')
        # _runner = TBME_SpeedRunner(filename='input.xml')
        # _runner = TBME_SpeedRunner(filename='input_B1.xml')
        # _runner.run()
    
        # _runner = TBME_SpeedRunner(filename='input_test_frD.xml', verbose=False)
        # _runner = TBME_SpeedRunner(filename='final_input.xml', verbose=False)
        # _runner.run()
        
        files_ = ['input_B1.xml', 'input_D1S.xml', 'input_M3Y_P2.xml', 'input_M3Y_P6.xml']
        for f in files_:
            _runner = TBME_SpeedRunner(filename=f)
            _runner.run()
            del _runner
    
    print(" [2B_MatrixElements.main] Program ended without incidences! Bye.")