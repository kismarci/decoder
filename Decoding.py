# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:27:53 2018

@author: Work
"""
# -*- coding: utf-8 -*-

import numpy as np
import rpy2.robjects as robjects
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from random import randrange
from scipy.stats import poisson
from functools import reduce

#######-------------------GET THE DATA AND CONSTRUNCT ALL THE VARIABLES------------------------------------------------ 
#for explanation see Replay.Rmd
f_ = open('replay_df.obj', 'rb')
dataframe = pickle.load(f_)  # variables come out in the order you put them in
f_.close()
pos=dataframe.iloc[0]['pos'] #matrix of time in sec and corresponding pos in meter
spt=dataframe.iloc[0]['spt']
actrunsleft=dataframe.iloc[0]['actrunsleft'] #cell 121: contains the time spent in each tiral at each position bin
actrunsright=dataframe.iloc[0]['actrunsright']
xmids=dataframe.iloc[0]['xmids']

ratemapsleft_t=actrunsleft[0:120][:][:].sum(axis=2)
Tmap_left=actrunsleft[120][:][:].sum(axis=1)

ratemapsright_t=actrunsright[0:120][:][:].sum(axis=2)
Tmap_right=actrunsright[120][:][:].sum(axis=1)
#ratemapsleftall = np.divide(ratemapsleft_t.T,Tmap_left)
data=robjects.r['load']("replay2.RData")
#print(list(data))
icellactive=robjects.r['i.cells.active'] #vagy jobb vagy balra futasnal volt aktiv, mivel index ezert ha be
icellactive=np.array(icellactive)#helyettesitem valahova 1-et ki kell vonni
icellactive_left=robjects.r['i.cells.active.left']
icellactive_left=np.array(icellactive_left)
icellactive_right=robjects.r['i.cells.active.right']
icellactive_right=np.array(icellactive_right)
 
Ncellsactiveleft =len(icellactive_left)
Ncellsactiveright = len(icellactive_right)

ratemaps_left=robjects.r['ratemaps.left']
ratemaps_left=np.array(ratemaps_left)
ratemaps_right=robjects.r['ratemaps.right']

robjects.r['load']("ratemaps_active.RData")
ratemaps_left_act=robjects.r['ratemaps.left']
ratemaps_left_act=np.array(ratemaps_left_act)
ratemaps_right_act=robjects.r['ratemaps.right']
ratemaps_right_act=np.array(ratemaps_right_act)

#activity colormap---------------------------------------------------------------------
#plt.close(fig)
i_cell=random.choice(icellactive_left) #32 position and 42 trials per each cell
#i_cell=20
fig, ax = plt.subplots()
img=ax.imshow((actrunsleft[i_cell-1,:,:] / actrunsleft[120,:,:]).T)
plt.gca().invert_yaxis()
ax.set_xticks(np.arange(0,len(xmids),3))
ax.set_xticklabels(xmids[::3])
plt.xticks(fontsize=8, rotation=45)
i_idx = np.where(icellactive_left==i_cell)[0]
plt.plot(ratemaps_left_act[:,i_idx]) 
#--------------------------------------------------------------------------------
plt.figure()
plt.plot(xmids,ratemaps_left_act[:,i_idx])
plt.xlabel('pos')
plt.ylabel('firing rate [Hz]')

plt.figure(2)
spikecount_left_act=actrunsleft[icellactive_left-1][:][:].sum(axis=2).T #contains the spike count for every active cell
# on the left run in each position
plt.scatter(xmids,spikecount_left_act[:,i_idx])

#------------Decoding based on toltal spike counts-------------------------------------------------
def decode_pois(lambda_P, r):
  # lambda: stimulus dependent firing rate
  # r: observed spike count
  post = poisson.pmf(r,lambda_P)
  normalised_post = post / sum(post)
  return normalised_post

cell=80
cell_idx = np.where(icellactive_left==cell)[0]
lambda_p=spikecount_left_act[:,cell_idx]
idx_pos = (np.abs(xmids -0.6)).argmin()
post=decode_pois(lambda_p , spikecount_left_act[idx_pos,cell_idx]) #response given to a position
plt.plot(xmids,ratemaps_left_act[:,cell_idx])
plt.xlabel('pos')
plt.ylabel('firing rate [Hz]')
plt.plot(xmids,post*10)

plt.figure(3)
cell_idx = np.where(icellactive_left==cell)[0]
lambda_p=spikecount_left_act[:,cell_idx]
idx_pos = (np.abs(xmids -0.6)).argmin()
post=decode_pois(lambda_p , spikecount_left_act[idx_pos,cell_idx]) #response given to a position
plt.scatter(xmids,spikecount_left_act[:,cell_idx]/10)
plt.plot(xmids,post*10)

plt.figure(4)
cell=80
cell_idx = np.where(icellactive_left==cell)[0]
lambda_p=spikecount_left_act[:,cell_idx]
decoded_pos=[]
for ipos in xmids:
    idx_pos = (np.abs(xmids - (ipos))).argmin()
    post=decode_pois(lambda_p , spikecount_left_act[idx_pos,cell_idx])
    decoded_pos.append(post.max())    
    if ipos>(0.4) and ipos<0.8:
        plt.scatter(xmids,spikecount_left_act[:,cell_idx]/10)
        plt.plot(xmids,post*10, label='%s' % ipos)
plt.legend()
decoded_pos=xmids[decoded_pos.index(max(decoded_pos))]

#------------Decoding based on triales-------------------------------------------------
actrunsleft_active=actrunsleft[icellactive_left-1][:][:]
random_indexes = random.sample(range(0,42),35)
training_set=actrunsleft_active[:,:,random_indexes]
test_set=actrunsleft_active[:,:,list(set(range(0,42))-set(random_indexes))]

#1. METHOD: calculate with spike counts, with one cell
lambda_sp_mean=training_set.mean(axis=2)
def decode_pois(lambda_P, r): #r csak egész szám lehet!!!!!
  # r: observed spike count
  post = poisson.pmf(r,lambda_P)
  normalised_post = post / sum(post)
  return normalised_post

plt.figure('sp count')
cell=9
cell_idx = np.where(icellactive_left==cell)[0]
plt.plot(xmids,lambda_sp_mean[cell_idx,:].T, label='sp mean')
#plt.plot(xmids,ratemaps_left_act[:,cell_idx]/10,label='firing rate')

idx_pos = (np.abs(xmids -1.2)).argmin()
#trial_idx=2
for trial_idx in range(np.shape(test_set)[2]):
    post=decode_pois(lambda_sp_mean[cell_idx,:].T , test_set[cell_idx,idx_pos,trial_idx]) #response given to a position
    plt.plot(xmids,post*10,label='trial %d post' %trial_idx)
plt.legend()

#It can be seen that if I choose a trial from the test set that has a spike number in a given position
#that is close to the mean sp number than the posterior will have a nice peak at the positon, however to find
#such a trial is quite unlikely

#2. MEHTOHD: calculate with the firing rate: spike number in a trial divided by the time spent in each tr.
# in each location that is (NAIVELY) the total time in a loc./number of trials (42) 
timeof1trial=Tmap_left/42
lambda_sp_mean_frek=np.divide(lambda_sp_mean,timeof1trial.T)

plt.figure('mean freq2')
cell=73
#trial_idx=2
cell_idx = np.where(icellactive_left==cell)[0]
idx_pos = (np.abs(xmids -1.2)).argmin()
for trial_idx in range(np.shape(test_set)[2]):
    response=np.round(test_set[cell_idx,idx_pos,trial_idx]/timeof1trial[idx_pos])
    post=decode_pois(lambda_sp_mean_frek[cell_idx,:].T , response) #response given to a position
    plt.plot(xmids,post*10,label='trial %d post' %trial_idx)
#plt.plot(xmids,ratemaps_left_act[:,cell_idx],label='firing rate')
plt.plot(xmids,lambda_sp_mean_frek[cell_idx,:].T,'--', label='sp mean freq',color='pink')
plt.legend()

#3. METHOD: decoding with multiple cell based on the mean firing rate
def decode_pois_2cells(lambda_P1,lambda_P2, r1,r2):
  # lambda: position dependent mean firing rate
  # r: observed spike count
  post = poisson.pmf(r1,lambda_P1)*poisson.pmf(r2,lambda_P2)
  normalised_post = post / sum(post)
  return normalised_post

#cells with similar place fields: (9,73,53), (44,78,61), (6,80)
plt.figure('mean freq 2 cells2')
#cell1=9
#cell2=80
cell1=61
cell2=78
cell_idx1 = np.where(icellactive_left==cell1)[0]
cell_idx2 = np.where(icellactive_left==cell2)[0]
idx_pos = (np.abs(xmids -0.2)).argmin()
for trial_idx in range(np.shape(test_set)[2]):
    response1=np.round(test_set[cell_idx1,idx_pos,trial_idx]/timeof1trial[idx_pos])
    response2=np.round(test_set[cell_idx2,idx_pos,trial_idx]/timeof1trial[idx_pos])
    post=decode_pois_2cells(lambda_sp_mean_frek[cell_idx1,:].T,lambda_sp_mean_frek[cell_idx2,:].T ,
                            response1, response2) 
    plt.plot(xmids,post*10,label='trial %d post' %trial_idx)
#plt.plot(xmids,ratemaps_left_act[:,cell_idx],label='firing rate')
plt.plot(xmids,lambda_sp_mean_frek[cell_idx1,:].T,'--', label='mean freq cell%d' %cell1,color='pink')
plt.plot(xmids,lambda_sp_mean_frek[cell_idx2,:].T,'--' ,label='mean freq cell%d' %cell2,color='purple')
plt.legend()

"""
-külön kell szedni hogy a traing setben osszeset mennyi idot tartozkodott egy pozicioban és a testsetben
nem pedig osszes ido/42
-a repsonse az idobol jojjon, nem 32 bin lesz egy futasban hanem mondjuk 0.1 sec-es binek, ekkor a lambdat
be kell szorozni a bin merettel, mert a poisson(lamda,k) az t=1 sec-re adja meg a valoszinuseget
tehát poisson(lambda*delta_t,k) lesz 
-leave one out cross validation, tehát 1 trial lesz a testset viszont végigmegyek az egész trial halmazon
mindegyik trial lesz egyszer test és az osszesbol maximum likelihoodot szamolunk a prediktalt hely és a
valodi hely kozotti kulonbségből vagy MAP-et, ezzel jellemezve a dekóder teljesitmenyet
a dekoder parametereit valtoztatva például a time bint, sejtek szamat figyelhetjuk a teljesitmenyt
mennyire bias-d a dekoder? a valodi helyhez kepest jobbra vagy balra prediktal e inkabb illetve mekkora a szorasa?
vegyuk bele a prior tudasunkat a pozicioral
terben folytonossagra atteres
"""

d=robjects.r['load']("actrunslefttime.RData")
actrunslefttimes=robjects.r['act.runs.left.time']
actrunslefttime=np.array(actrunslefttimes[1]) # matrix with the spikes of each cell on each time frame with 0.1 sec wide time bins
actrunslefttime=actrunslefttime[icellactive_left-1]
nframes=np.array(actrunslefttimes[2])#number of dataframes in each trial
pos_of_frame=np.array(actrunslefttimes[0])
pos_of_frame_list=[]#list of the positions for trials in each time bin
actrunslefttime_list=[]#list of matrices with the spikes of each cell on each time frame with 0.1 sec wide time bins 
nframes=nframes.astype(int)
sz=0
ssz=nframes[0]
i=0
for i_trial in nframes:
    if i==0:
        M=actrunslefttime[:,sz:(sz+i_trial)].copy()
        P=pos_of_frame[sz:(sz+i_trial)].copy()
        actrunslefttime_list.append(M)
        pos_of_frame_list.append(P)
    else:
        M=actrunslefttime[:,ssz:(ssz+i_trial)].copy()
        P=pos_of_frame[ssz:(ssz+i_trial)].copy()
        actrunslefttime_list.append(M)
        pos_of_frame_list.append(P)
        ssz=ssz+i_trial
    i=i+1

#4. METHOD: decoding with multiple cell based on the mean firing rate, splitting the time spent in training and in test set
#np.diff(pos[[4363-1,4799-1],0])
#times=[] #contains the times of the individual left runs
#for i in range(42):
#    times.append(np.diff(pos[[int(ileftruns[i,0])-1,int(ileftruns[i,1])-1],0]))    
# 
timeof1trial_trainingset=actrunsleft[120,:,random_indexes].mean(axis=0)
timeof1trial_testset=actrunsleft[120,:,list(set(range(0,42))-set(random_indexes))].mean(axis=0)
lambda_sp_mean_frek_splitted=np.divide(lambda_sp_mean,timeof1trial_trainingset.T)
lambda_sp_mean_frek_splitted+=10**-3
test_indices=list(set(range(0,42))-set(random_indexes))

plt.figure('mean freq 2 cells2 splitted')
cell1=61
cell2=78
cell_idx1 = np.where(icellactive_left==cell1)[0]
cell_idx2 = np.where(icellactive_left==cell2)[0]
idx_pos = (np.abs(xmids -0.2)).argmin()
for trial_idx in range(np.shape(test_set)[2]):
    response1=np.round(test_set[cell_idx1,idx_pos,trial_idx]/timeof1trial_testset[idx_pos])
    response2=np.round(test_set[cell_idx2,idx_pos,trial_idx]/timeof1trial_testset[idx_pos])
    post=decode_pois_2cells(lambda_sp_mean_frek_splitted[cell_idx1,:].T,
                            lambda_sp_mean_frek_splitted[cell_idx2,:].T ,response1, response2) 
    plt.plot(xmids,post*10,label='trial %d post' %trial_idx)
#plt.plot(xmids,ratemaps_left_act[:,cell_idx],label='firing rate')
plt.plot(xmids,lambda_sp_mean_frek_splitted[cell_idx1,:].T,'--', label='mean freq cell%d' %cell1,color='pink')
plt.plot(xmids,lambda_sp_mean_frek_splitted[cell_idx2,:].T,'--' ,label='mean freq cell%d' %cell2,color='purple')
plt.legend()

#responses extract from the timeserie insted of the spike/positon
frame=28
for trial_idx in test_indices:
    response1=np.round(actrunslefttime_list[trial_idx][cell_idx1,frame]/0.1)
    response2=np.round(actrunslefttime_list[trial_idx][cell_idx2,frame]/0.1)
    post=decode_pois_2cells(lambda_sp_mean_frek_splitted[cell_idx1,:].T*0.1,
                            lambda_sp_mean_frek_splitted[cell_idx2,:].T*0.1 ,response1, response2) 
    plt.plot(xmids,post*10,label='trial %d post' %trial_idx)
#plt.plot(xmids,ratemaps_left_act[:,cell_idx],label='firing rate')
plt.plot(xmids,lambda_sp_mean_frek_splitted[cell_idx1,:].T,'--', label='mean freq cell%d' %cell1,color='pink')
plt.plot(xmids,lambda_sp_mean_frek_splitted[cell_idx2,:].T,'--' ,label='mean freq cell%d' %cell2,color='purple')
plt.axvline(x=pos_of_frame_list[trial_idx][frame],label='real position',color='black',ls=':')
plt.legend()

#Decoding with N cells
def decode_pois_Ncells(lambda_PN,rN,cells):
  # lambdaN:matrix of position dependent mean firing rates
  # r:vektor observed firing rate in the given time bin (spike count/time in the bin)
  #cells: vektor of N cells
  postvec=[]
  for icell in cells:
      postvec.append(poisson.pmf(rN[icell],lambda_PN[:,icell]))
  post = reduce(lambda x, y: x*y, postvec)
  normalised_post = post / (sum(post))
  return normalised_post

def decode_pois_Ncells(lambda_PN,rN,cells, *prior):
  # lambdaN:matrix of position dependent mean firing rates
  # r:vektor observed firing rate in the given time bin (spike count/time in the bin)
  #cells: vektor of N cells
  postvec=[]
  for icell in cells:
      postvec.append(np.log(poisson.pmf(rN[icell],lambda_PN[:,icell])))
  for p in prior: postvec.append(p)
  post = reduce(lambda x, y: x+y, postvec)
  ppost=np.power(np.e,post)
  normalised_post = ppost / (sum(ppost))
  return normalised_post


cells=random.sample(range(0,54),30)
frame=32
trial_idx=random.choice(test_indices)
responseN=np.round(actrunslefttime_list[trial_idx][:,frame]/0.1)
prior_pos=timeof1trial_trainingset/sum(timeof1trial_trainingset)
post=decode_pois_Ncells(lambda_sp_mean_frek_splitted.T*0.1,responseN,cells, prior_pos) 
plt.plot(xmids,post*10,label='trial %d post' %trial_idx)
plt.axvline(x=pos_of_frame_list[trial_idx][frame],label='real position in trial %d' %trial_idx,color='black',ls=':')
plt.legend(prop={'size':10})
predicted_pos=xmids[post.argmax()]

#assesing the decoder's permormance by leave-one-out crossvalidation
import scipy

def cross_validator(decoder,cell_number,inc_prior='no'):
    m=[]
    #ent=[]
    run=0
    while run<5:
        error=[]
        #ent_k=[]
        cellscv=random.sample(range(0,54),cell_number)
        for k in range(42):
            error_f=[]
            #entr_f=[]
            indexes=list(range(42))
            indexes.remove(k)
            training_setcv=actrunsleft_active[:,:,indexes]
            lambda_sp_meancv=training_setcv.mean(axis=2)
            timeof1trial_trainingsetcv=actrunsleft[120,:,indexes].mean(axis=0)
            lambda_sp_mean_frek_splittedcv=np.divide(lambda_sp_meancv,timeof1trial_trainingsetcv.T)
            lambda_sp_mean_frek_splittedcv+=10**-2
            prior_poscv=timeof1trial_trainingsetcv/sum(timeof1trial_trainingsetcv)
            for f in range(nframes[k]):
                responseNcv=np.round(actrunslefttime_list[k][:,f])
                if inc_prior=='yes':
                    postcv=decoder(lambda_sp_mean_frek_splittedcv.T*0.1,responseNcv,cellscv,prior_poscv)
                else: postcv=decoder(lambda_sp_mean_frek_splittedcv.T*0.1,responseNcv,cellscv)
                predicted_pos=xmids[postcv.argmax()]
                #entr_f.append(scipy.stats.entropy(postcv))
                error_f.append((predicted_pos-pos_of_frame_list[k][f])) #**2)
            #ent_k.append(np.mean(entr_f))            
            error.append(np.mean(error_f))
        mean_error=np.mean(error)
        #ent.append(np.mean(ent_k))
        m.append(mean_error)
        run += 1
    m_mean_error=np.mean(m)
    #ent_mean=np.mean(ent)
    sd_error=np.std(m)
    return (m_mean_error, sd_error,m)
    #return(ent_mean,postcv)

    
cross_validator(decoder=decode_pois_Ncells,cell_number=10)
#calculates the mean error of the predicted pos from the real pos by averaging the errors of all the frame and all the trials
#likelihoodok helyett loglikelihoodot kell venni es azok osszeget a szorzat helyett mert a szorzatban a numerikus hibak felgyulemlenek

#we take into account the prior knowledge of the position
cross_validator(decoder=decode_pois_Ncells,cell_number=10,inc_prior='yes')



plt.scatter(list(range(32)),lambda_sp_mean_frek[10,:])
plt.plot(lambda_sp_mean_frek[10,:])
plt.scatter(xmids,lambda_sp_mean_frek_splitted[10,:])

from scipy.interpolate import interp1d
x=xmids
y=lambda_sp_mean_frek_splitted[10,:]
f = interp1d(x, y, kind='cubic')

xnew = np.linspace(min(xmids), max(xmids), num=1000, endpoint=True)
plt.plot(x, y, 'o', xnew, f(xnew), '-')
plt.legend(['data', 'cubic'], loc='best')
plt.show()
































