

from numpy import load,int64,max,mean,sort
from os import listdir

from rbf import rbf


from sys import path,argv,exit

from utils import rms
 


# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint

Sets='data/MOGA_DATA_136i_os.npz.npz'
realdata_validations = load(Sets)['VA_SET']
realdata_testing = load(Sets)['TE_SET']
realdata_training = load(Sets)['TR_SET']


f=open('allmodels_tables/allmodels.csv','w')

f.write("Model_number\tMin_Scale_V\tMax_Scale_V\tMean_Scale_V\tRMS_Scale_V\tMin\tMax\tMean\tRMS\tMin_Scale_tt\tMax_Scale_tt\tMean_Scale_tt\tRMS_Scale_tt\tMin_Scale_tr\tMax_Scale_tr\tMean_Scale_tr\tRMS_Scale_tr\n")

models_numbers=len(listdir('allmodels/'))
count=0

for x in listdir('allmodels/'):
    try:
        if not x.split('.')[-1] == 'npz':
            print "Only npz files allowed"
            exit(-1)
    except:
        print "Only npz files allowed"
        exit(-1)
        
    count+=1   
    print 'file ',x,' counting ',count,' / ',models_numbers
    
    modelnumber = x.split('/')[-1].split('.')[0]

    x = load("allmodels/"+ modelnumber + ".npz")

    neurons = x['spreads'].shape[0]

    MRT_INPUT_LAGS = int64(sort(x['features']))

    MRTmodel = rbf(None,None,neurons=neurons,bias=True,afun='gauss',iC='random')

    MRTmodel.C = x['centers']
    MRTmodel.S = x['spreads'].flatten()
    MRTmodel.W = (x['weights'].flatten(),0)
    
    Yvalidation = MRTmodel.ProduceOutput(realdata_validations[:, MRT_INPUT_LAGS])
    Ytesting = MRTmodel.ProduceOutput(realdata_testing[:, MRT_INPUT_LAGS])
    Ytraining = MRTmodel.ProduceOutput(realdata_training[:, MRT_INPUT_LAGS])

    vEscalevalidation = realdata_validations[:,0]-Yvalidation
    vEscaletesting = realdata_testing[:,0]-Ytesting
    vEscaletraining = realdata_training[:,0]-Ytraining

    vE = realdata_validations[:,0] - Yvalidation
    
    dados = []

    dados.insert(len(dados),min(vEscalevalidation))
    dados.insert(len(dados),max(vEscalevalidation))
    dados.insert(len(dados),mean(vEscalevalidation))
    dados.insert(len(dados),rms(vEscalevalidation))

    dados.insert(len(dados),min(vE))
    dados.insert(len(dados),max(vE))
    dados.insert(len(dados),mean(vE))
    dados.insert(len(dados),rms(vE))

    dados.insert(len(dados),min(vEscaletesting))
    dados.insert(len(dados),max(vEscaletesting))
    dados.insert(len(dados),mean(vEscaletesting))
    dados.insert(len(dados),rms(vEscaletesting))

    dados.insert(len(dados),min(vEscaletraining))
    dados.insert(len(dados),max(vEscaletraining))
    dados.insert(len(dados),mean(vEscaletraining))
    dados.insert(len(dados),rms(vEscaletraining))

    
    tmpwrite = "".join("\t" + "%s" % str(x) for x in dados) # .replace('.',',')
    
    f.write("""%s%s\n""" % (modelnumber,tmpwrite))

   
f.close()

print '- finish -'
