# Run file by itself (or import it?) Comment out following code if building 
# from command line. (setup default)
import sys
old_sys_argv = sys.argv[:]
sys.argv = ['', 'build_ext', '--inplace']

import envs.rover_domain_cython.rover_domain_setup

sys.argv = old_sys_argv

from rover_domain import *