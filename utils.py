'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-03-29
	content: 
'''
import glob
import natsort
import os.path as osp
import os
import re

def get_work_dir(run_id:str):
    ''' get the full work dir (name+index) give the name '''
    all_runs = glob.glob(run_id+'*')
    all_runs = [run for run in all_runs if re.match(run_id+'_*\d+', run)]
    all_runs = natsort.natsorted(all_runs)
    if all_runs:
        run_id_cnt = re.findall('_\d+', all_runs[-1])
        run_id_cnt = int(run_id_cnt[-1][1:])
        run_id  = run_id + '_' + str(run_id_cnt+1)
    else:
        run_id = run_id + '_0'
    os.mkdir((run_id))
    return run_id