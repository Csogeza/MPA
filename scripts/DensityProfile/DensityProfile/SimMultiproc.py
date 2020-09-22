from tardis import run_tardis
from multiprocessing import Pool
from BPLDens import model_saver as ms
import os
import pandas as pd

def run_sim(fn, drs = 'SimFiles/'):
	thvel = fn.split('_')[-1][:-4]
	sim = run_tardis(drs+fn)
	ms.dump_to_hdf(sim, '/afs/mpa/data/csogeza/SimResults/sim_vth_'+thvel+'.hdf')
	

filelist = [i for i in os.listdir('SimFiles') if i.startswith('model')]

if __name__ == '__main__':
	p = Pool(8)
	p.map(run_sim, filelist)
