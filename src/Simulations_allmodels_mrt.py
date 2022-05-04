#!/usr/bin/python
# -*- coding: utf-8 -*-
#

from numpy import load, int64,max,mean,sort
from os import listdir
import pylab
from rbf import rbf

from sys import path,argv,exit

#LOCAL_LIB_PATH="/home/sergiosilva/Documents/ualg/clusterlib/trunk"
#path.append(LOCAL_LIB_PATH)

from utils import rms


MRT_minV=10.0
MRT_maxV=35.0

def unscale(x,m,M):
  return x*(M-m)+m

# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint

realdata_validations = load('data/Data_Final_heating_random_mrt.npz')['Data_validation']

pylab.ion()
pylab.figure(figsize=(12,8))
pylab.show()

for x in listdir('allmodels_mrt/'):
    try:
        if not x.split('.')[-1] == 'npz':
            print "Only npz files allowed"
            exit(-1)
    except:
        print "Only npz files allowed"
        exit(-1)

    print x

    modelnumber = x.split('/')[-1].split('.')[0]

    x = load("allmodels_mrt/"+ modelnumber + ".npz")

    neurons = x['spreads'].shape[0]

    MRT_INPUT_LAGS = int64(sort(x['features']))

    MRTmodel = rbf(None,None,neurons=neurons,bias=True,afun='gauss',iC='random')

    MRTmodel.C = x['centers']
    MRTmodel.S = x['spreads'].flatten()
    MRTmodel.W = (x['weights'].flatten(),0)
    
    Y = MRTmodel.ProduceOutput(realdata_validations[:, MRT_INPUT_LAGS])

    vEscale = realdata_validations[:,0]-Y
    vE = unscale(realdata_validations[:,0],MRT_minV,MRT_maxV) - unscale(Y,MRT_minV,MRT_maxV)

    print "max of error unscale ", max(vE)
    print "min of error unscale", min(vE)
    print "mean of error unscale", mean(vE)

    print "root mean square unscale", rms(vE)

    print "max of error scale", max(vEscale)
    print "min of error scale", min(vEscale)
    print "mean of error scale", mean(vEscale)

    print "root mean square scale", rms(vEscale)
    
    pylab.clf()

    pylab.title('Target Validation Data and Model Output Data')
    real_legend = pylab.plot(unscale(realdata_validations[:,0],MRT_minV,MRT_maxV),label='Real')
    model_legend = pylab.plot(unscale(Y,MRT_minV,MRT_maxV),label='Model Output')
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=6,ncol=2, mode="expand", borderaxespad=0.)

    pylab.ylabel('Mean Radiant Temperature')
    pylab.xlabel('Number of points')
    ax = pylab.draw()

    print vE
    _breakpoint()  # 671d6f67


