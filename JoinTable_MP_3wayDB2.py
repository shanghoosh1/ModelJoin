import time
from multiprocessing import Queue, Process, current_process, Manager
from multiprocessing import Queue
import pandas as pd
import numpy as np
import json
import DMDN_onClusters as learner
############################################################################################## 0 hyperparameters
t=time.time()
embed_use=True
num_clusters1=50
num_joinSamples=100000
Num_proc=37
############################################################################################## 1 fetch frequecies of join attributes
# with open('embeddings/Freq_web_sales_customers.txt', 'r') as file:
#     freq_WS_customers=(json.load(file))
# with open('embeddings/Freq_store_sales_customers.txt', 'r') as file:
#     freq_SS_customers=(json.load(file))
# with open('embeddings/Freq_store_sales_items.txt', 'r') as file:
#     freq_SS_items=(json.load(file))
# with open('embeddings/Freq_catalog_sales_items.txt', 'r') as file:
#     freq_CS_items=(json.load(file))

with open('clusters/DB2/freq_tbl2_att1.txt', 'r') as file:
    jTBL0_1=(json.load(file))
with open('clusters/DB2/freq_tbl3_att0.txt', 'r') as file:
    jTBL1_0=(json.load(file))
with open('clusters/DB2/freq_tbl3_att1.txt', 'r') as file:
    jTBL1_1=(json.load(file))
with open('clusters/DB2/freq_tbl0_att0.txt', 'r') as file:
    jTBL2_0=(json.load(file))

############################################################################################## 2 fetch necessary embeddings
# table="embeddings/store_sales_customers_items_stores_embedings.txt"
table="embeddings/DB2/tbl_3_att0_embed.txt"
embedings = pd.read_csv(table,delimiter=',',header=None).values
M1_embedings={}
for i in range(len(embedings[:,0])):
    M1_embedings.update({str(int(embedings[i,0])):tuple(embedings[i,1:])})
embedings=None

################################################################################################ 3 frequency of each x per cluster
# import json
# with open('embeddings/clus_dict_100c.txt', 'r') as file:
#     clus_dict=(json.load(file))
# with open('embeddings/freq_dict_100c.txt', 'r') as file:
#     freq_dict=(json.load(file))

import json
with open('clusters/DB2/50c/tbl3_att1clust_dict50.txt', 'r') as file:
    clus_dict=(json.load(file))
with open('clusters/DB2/50c/tbl3_att1freq_dict50.txt', 'r') as file:
    freq_dict=(json.load(file))
################################################################################################ 5 Dynamic programming
add = 'clusters/DB2/50c/models/model_tbl3_'
out_add='DB2/server_t2_t3_t0/'
manager = Manager()
outList = manager.list()
def dyn_prog_MP(tasks_to_accomplish,clus_dict,freq_dict,M1_embedings ):
    t = time.time()
    cluster_model1=[[] for i in range(num_clusters1)]


    # for i in range(num_clusters1):
    #     myDMDN = learner.DMDN(no_inBins=100,no_layers=5,no_hidden=500)
    #     myDMDN.restore_ordinalY('./model/model_cluster' + str(i))
    #     myDMDN.useEmbeding=True
    #     myDMDN.buildmodel()
    #     myDMDN.restore('./model/model_cluster' + str(i))
    #     cluster_model1[i]=myDMDN


    for i in range(num_clusters1):
        model = learner.DMDN(no_layers=5, no_hidden=200)
        model.restore_ordinalY(add + str(i))
        no_out_bins = len(model.ordinalYencoding.main_dict)
        model.buildModel_largeData(no_out_bins, M1_embedings)
        model.restore(add + str(i))
        cluster_model1[i] = model

    print('time for loading models: '+str(time.time()-t))
    t=time.time()
    cnt=0
    while True:
        if tasks_to_accomplish.empty():
            print('queue is empty ######################################################################')
            break
        cnt += 1
        try:
             task = tasks_to_accomplish.get()
        except:
            print('########################################## ' + str(current_process().name)+' got error   #####################################################################')
            break

        key=task[0]
        value = task[1]
        y_probs = {}
        clu = list(clus_dict[str(key)])
        freq = list(freq_dict[str(key)])
        for i, inx in enumerate(clu):
            m =cluster_model1[inx]
            probs = m.predict_one(list(M1_embedings[str(key)]), freq[i])
            y_probs = {**y_probs, **probs}
        summ = 0
        sumYprobs = sum(y_probs.values())
        for k in y_probs.keys():
            if k in D1_on_z1:
                summ += y_probs[k]/sumYprobs * D1_on_z1[k]
        print('done' + str(cnt) + ' with '+ str(current_process().name))
        # tasks_that_are_done.put({key: float(value) * summ})
        outList.append([key,float(value) * summ])
    return True
ttt=time.time()
tt =time.time()
D1={z:jTBL1_1[z] *jTBL2_0[z]  for z in jTBL1_1.keys() if z in jTBL2_0}
time_D1=time.time()-tt
tt=time.time()
D1_on_z1={z:D1[z] /jTBL1_1[z]  for z in D1.keys() if z in jTBL1_1}
D2_y1y2={z:jTBL1_0[z] *jTBL0_1[z] for z in jTBL1_0.keys() if z in jTBL0_1}
number_of_task = len(D2_y1y2)
# number_of_task=10
number_of_processes = Num_proc
tasks_to_accomplish = Queue()
processes = []
aa=0
for tsk in D2_y1y2.items():
    tasks_to_accomplish.put(tsk)
    aa+=1
    if aa>number_of_task:
        break
# creating processes
for w in range(number_of_processes):
    p = Process(target=dyn_prog_MP, args=(tasks_to_accomplish, clus_dict,freq_dict,M1_embedings))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
print('time for dynamic programming: '+str(time.time()-tt))
time_D2=time.time()-tt
time_D1D2=time.time()-ttt
# print the output
D2 = {}
print('size of out'+ str(len(outList)))
print('size of in'+ str(len(D2_y1y2)))
for l in outList:
    D2.update({l[0]:l[1]})
    # print(l)
# while not tasks_that_are_done.empty():
#     q=tasks_that_are_done.get()
#     D2.update(q)
#     print(str(q))

#################################################################################################### 6 generate samples
outList = manager.list()
t_list=manager.list()
def skeletonGeneration(tasks_to_accomplish,clus_dict,freq_dict,M1_embedings ):
    t = time.time()
    cluster_model1=[[] for i in range(num_clusters1)]

    for i in range(num_clusters1):
        model = learner.DMDN(no_layers=5, no_hidden=200)
        model.restore_ordinalY(add + str(i))
        no_out_bins = len(model.ordinalYencoding.main_dict)
        model.buildModel_largeData(no_out_bins, M1_embedings)
        model.restore(add + str(i))
        cluster_model1[i] = model
    print('time for loading models: '+str(time.time()-t))
    t_list.append('loadingTime_'+str(current_process().name)+': '+ str(time.time()-t))
    t=time.time()
    cnt=0

    t_l=time.time()
    while True:
        if tasks_to_accomplish.empty():
            print('queue is empty ######################################################################')
            break
        cnt += 1
        try:
             task = tasks_to_accomplish.get()
        except:
            print('########################################## ' + str(current_process().name)+' got error   #####################################################################')
            break
        z_probs = {}
        y_probs = {}
        clu = list(clus_dict[str(task[-1])])
        freq = list(freq_dict[str(task[-1])])
        for i, inx in enumerate(clu):
            probs = cluster_model1[inx].predict_one(list(M1_embedings[str(task[-1])]), freq[i])
            y_probs = {**y_probs, **probs}
        sumP = 0
        for yy in y_probs.keys():
            # aa=y_probs[yy] * D1[yy]
            aa = y_probs[yy] * D1_on_z1[yy]
            sumP += aa
            z_probs.update({yy: aa})
        pp = [h / sumP for h in z_probs.values()]
        sample=np.random.choice(list(z_probs.keys()), 1, p=pp)
        task.append(sample[0])
        outList.append(task)
        print('done' + str(cnt) + ' with ' + str(current_process().name))
        if len(outList)%1000==0:
                t_list.append(str(len(outList))+', '+str(current_process().name)+', '+str(time.time()-t_l))

    return True

ttt=time.time()
tt =time.time()
aa=sum(D2.values())
P={a:D2[a]/aa for a in D2.keys()}
D2_samples=np.random.choice(list(D2.keys()), num_joinSamples, p=list(P.values()))
D1_samples=[]
time_sample_D1=time.time()-tt
tt=time.time()
# number_of_task = len(D2_samples)
number_of_task=num_joinSamples
number_of_processes = Num_proc
tasks_to_accomplish = Queue()
processes = []
aa=0
for tsk in D2_samples:
    tasks_to_accomplish.put([tsk])
    aa+=1
    if aa>number_of_task:
        break
# creating processes
for w in range(number_of_processes):
    p = Process(target=skeletonGeneration, args=(tasks_to_accomplish, clus_dict,freq_dict,M1_embedings))
    processes.append(p)
    p.start()

# completing process
for p in processes:
    p.join()
print('time for generating is: '+str(time.time()-tt))
time_sampleD2=time.time()-tt
time_allSamples=time.time()-ttt
# print the output
# D1_samples=outList
# D1_samples=np.array(D1_samples).reshape(len(D1_samples),1)
# print(len(D1_samples))
# print(len(D2_samples))
#
# D2_samples=np.array(D2_samples).reshape(len(D2_samples),1)
# skeleton=np.append(D2_samples,D1_samples,axis=1)
# # skeleton=list(zip(D2_samples,D1_samples))
skeleton=outList
# for row in outList:
#     print(str(row))
skeleton=np.array(skeleton).reshape(len(skeleton),2)
np.savetxt(out_add+'sample.csv', skeleton.astype('str'), delimiter=",", fmt='%s, %s')
t=[time_D1,time_D2,time_D1D2,time_sample_D1,time_sampleD2,time_allSamples]
f=open(out_add+'time.txt','a+')
f.write(str(t))
f.write('\n')
for row in t_list:
    f.write(str(row)+'\n')
f.close()






