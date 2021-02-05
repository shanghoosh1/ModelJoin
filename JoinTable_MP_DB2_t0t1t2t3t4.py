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

with open('clusters/DB2/freq_tbl0_att1.txt', 'r') as file:
    jTBL0_1=(json.load(file))
with open('clusters/DB2/freq_tbl1_att0.txt', 'r') as file:
    jTBL1_0=(json.load(file))
with open('clusters/DB2/freq_tbl1_att1.txt', 'r') as file:
    jTBL1_1=(json.load(file))
with open('clusters/DB2/freq_tbl2_att0.txt', 'r') as file:
    jTBL2_0=(json.load(file))
with open('clusters/DB2/freq_tbl2_att1.txt', 'r') as file:
    jTBL2_1=(json.load(file))
with open('clusters/DB2/freq_tbl3_att0.txt', 'r') as file:
    jTBL3_0=(json.load(file))
with open('clusters/DB2/freq_tbl3_att1.txt', 'r') as file:
    jTBL3_1=(json.load(file))
with open('clusters/DB2/freq_tbl4_att0.txt', 'r') as file:
    jTBL4_0=(json.load(file))

############################################################################################## 2 fetch necessary embeddings
# table="embeddings/store_sales_customers_items_stores_embedings.txt"
table="embeddings/DB2/tbl_1_att0_embed.txt"
embedings = pd.read_csv(table,delimiter=',',header=None).values
M1_embedings={}
for i in range(len(embedings[:,0])):
    M1_embedings.update({str(int(embedings[i,0])):tuple(embedings[i,1:])})

table="embeddings/DB2/tbl_2_att0_embed.txt"
embedings = pd.read_csv(table,delimiter=',',header=None).values
M2_embedings={}
for i in range(len(embedings[:,0])):
    M2_embedings.update({str(int(embedings[i,0])):tuple(embedings[i,1:])})

table="embeddings/DB2/tbl_3_att0_embed.txt"
embedings = pd.read_csv(table,delimiter=',',header=None).values
M3_embedings={}
for i in range(len(embedings[:,0])):
    M3_embedings.update({str(int(embedings[i,0])):tuple(embedings[i,1:])})

################################################################################################ 3 frequency of each x per cluster

import json
with open('clusters/DB2/50c/tbl1_att1clust_dict50.txt', 'r') as file:
    clus_dict1=(json.load(file))
with open('clusters/DB2/50c/tbl1_att1freq_dict50.txt', 'r') as file:
    freq_dict1=(json.load(file))
with open('clusters/DB2/50c/tbl2_att1clust_dict50.txt', 'r') as file:
    clus_dict2=(json.load(file))
with open('clusters/DB2/50c/tbl2_att1freq_dict50.txt', 'r') as file:
    freq_dict2=(json.load(file))
with open('clusters/DB2/50c/tbl3_att1clust_dict50.txt', 'r') as file:
    clus_dict3=(json.load(file))
with open('clusters/DB2/50c/tbl3_att1freq_dict50.txt', 'r') as file:
    freq_dict3=(json.load(file))
################################################################################################ 5 Dynamic programming
add1 = 'clusters/DB2/50c/models/model_tbl1_'
add2 = 'clusters/DB2/50c/models/model_tbl2_'
add3 = 'clusters/DB2/50c/models/model_tbl3_'

out_add='DB2/server_t0_t1_t2_t3_t4/'
manager = Manager()
outList = manager.list()
def dyn_prog_MP(tasks_to_accomplish,clus_dict,freq_dict,M_embedings,add,D_on_z1 ):
    t = time.time()
    cluster_model1=[[] for i in range(num_clusters1)]
    for i in range(num_clusters1):
        model = learner.DMDN(no_layers=5, no_hidden=200)
        model.restore_ordinalY(add + str(i))
        no_out_bins = len(model.ordinalYencoding.main_dict)
        model.buildModel_largeData(no_out_bins, M_embedings)
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
            if k in D_on_z1:
                summ += y_probs[k]/sumYprobs * D_on_z1[k]
        print('done' + str(cnt) + ' with '+ str(current_process().name))
        # tasks_that_are_done.put({key: float(value) * summ})
        outList.append([key,float(value) * summ])
    return True
ttt=time.time()
tt =time.time()
D4={z:jTBL3_1[z] *jTBL4_0[z]  for z in jTBL3_1.keys() if z in jTBL4_0}
time_D1=time.time()-tt
tt=time.time()
D4_on_z1={z:D4[z] /jTBL3_1[z]  for z in D4.keys() if z in jTBL3_1}
D3_y1y2={z:jTBL3_0[z] *jTBL2_1[z] for z in jTBL3_0.keys() if z in jTBL2_1}
number_of_task = len(D3_y1y2)





#number_of_task=10




number_of_processes = Num_proc
tasks_to_accomplish = Queue()
processes = []
aa=0
for tsk in D3_y1y2.items():
    tasks_to_accomplish.put(tsk)
    aa+=1
    if aa>number_of_task:
        break
# creating processes
for w in range(number_of_processes):
    p = Process(target=dyn_prog_MP, args=(tasks_to_accomplish, clus_dict3,freq_dict3,M3_embedings,add3,D4_on_z1))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
time_D2=time.time()-tt
time_D1D2=time.time()-ttt
# print the output
D3 = {}
for l in outList:
    D3.update({l[0]:l[1]})
print('second att in daynamic prog finished')
outList = manager.list()
D3_on_z1={z:D3[z] /jTBL2_1[z]  for z in D3.keys() if z in jTBL2_1}
D2_y1y2={z:jTBL2_0[z] *jTBL1_1[z] for z in jTBL2_0.keys() if z in jTBL1_1}
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
    p = Process(target=dyn_prog_MP, args=(tasks_to_accomplish, clus_dict2,freq_dict2,M2_embedings,add2,D3_on_z1))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
D2 = {}
for l in outList:
    D2.update({l[0]:l[1]})
print('third att in daynamic prog finished')

D2_on_z1={z:D2[z] /jTBL1_1[z]  for z in D2.keys() if z in jTBL1_1}
D1_y1y2={z:jTBL1_0[z] *jTBL0_1[z] for z in jTBL1_0.keys() if z in jTBL0_1}
number_of_task = len(D1_y1y2)
# number_of_task=10
number_of_processes = Num_proc
tasks_to_accomplish = Queue()
processes = []
aa=0
for tsk in D1_y1y2.items():
    tasks_to_accomplish.put(tsk)
    aa+=1
    if aa>number_of_task:
        break
# creating processes
for w in range(number_of_processes):
    p = Process(target=dyn_prog_MP, args=(tasks_to_accomplish, clus_dict1,freq_dict1,M1_embedings,add1,D2_on_z1))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
D1 = {}
for l in outList:
    D1.update({l[0]:l[1]})
dynamicTime=time.time()-ttt
print('time for dynamic programming: '+str(time.time()-tt))
#################################################################################################### 6 generate samples
outList = manager.list()
t_list=manager.list()
def skeletonGeneration(tasks_to_accomplish,clus_dict,freq_dict,M_embedings,add,D_on_z1 ,att_id ):
    t = time.time()
    cluster_model1=[[] for i in range(num_clusters1)]

    for i in range(num_clusters1):
        model = learner.DMDN(no_layers=5, no_hidden=200)
        model.restore_ordinalY(add + str(i))
        no_out_bins = len(model.ordinalYencoding.main_dict)
        model.buildModel_largeData(no_out_bins, M_embedings)
        model.restore(add + str(i))
        cluster_model1[i] = model
    print('time for loading models: '+str(time.time()-t))
    t_list.append('loadingTime_'+str(current_process().name)+': '+ str(time.time()-t))
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
            if yy in D_on_z1:
                aa = y_probs[yy] * D_on_z1[yy]
                sumP += aa
                z_probs.update({yy: aa})



        pp = [h / sumP for h in z_probs.values()]
        sample=np.random.choice(list(z_probs.keys()), 1, p=pp)
        task.append(sample[0])
        outList.append(task)
        print('done' + str(cnt) + ' with ' + str(current_process().name))
        if len(outList)%1000==0:
                t_list.append(str(att_id)+', '+str(len(outList))+', '+str(time.time()-t_l))

    return True


aa=sum(D1.values())
P={a:D1[a]/aa for a in D1.keys()}
D1_samples=np.random.choice(list(D1.keys()), num_joinSamples, p=list(P.values()))
print('first att generated')

number_of_task=num_joinSamples
number_of_processes = Num_proc
tasks_to_accomplish = Queue()
processes = []
aa=0
for tsk in D1_samples:
    tasks_to_accomplish.put([tsk])
    aa+=1
    if aa>number_of_task:
        break

for w in range(number_of_processes):
    p = Process(target=skeletonGeneration, args=(tasks_to_accomplish, clus_dict1,freq_dict1,M1_embedings,add1,D2_on_z1,2))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
skeleton=outList
outList = manager.list()
#skeleton=np.array(skeleton).reshape(len(skeleton),2)
print('second att generated')


number_of_task=num_joinSamples
number_of_processes = Num_proc
tasks_to_accomplish = Queue()
processes = []
aa=0
for tsk in skeleton:
    print(tsk)
    tasks_to_accomplish.put(tsk)
    aa+=1
    if aa>number_of_task:
        break

for w in range(number_of_processes):
    p = Process(target=skeletonGeneration, args=(tasks_to_accomplish, clus_dict2,freq_dict2,M2_embedings,add2,D3_on_z1,3))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
skeleton=outList
outList = manager.list()
for row in skeleton:
    print(str(row)+'\n')

#skeleton=np.array(skeleton).reshape(len(skeleton),3)
print('third att generated')

number_of_task=num_joinSamples
number_of_processes = Num_proc
tasks_to_accomplish = Queue()
processes = []
aa=0
for tsk in skeleton:
    tasks_to_accomplish.put(tsk)
    aa+=1
    if aa>number_of_task:
        break

for w in range(number_of_processes):
    p = Process(target=skeletonGeneration, args=(tasks_to_accomplish, clus_dict3,freq_dict3,M3_embedings,add3,D4_on_z1,4))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
skeleton=outList
#skeleton=np.array(skeleton)
print('forth att generated')
skeleton=np.array(skeleton).reshape(len(skeleton),4)

np.savetxt(out_add+'sample.csv', skeleton.astype('str'), delimiter=",", fmt='%s, %s, %s, %s')

#f=open(out_add+'sample.csv','a+')
#for row in skeleton:
#    f.write(str(row)+'\n')
#f.close()

f=open(out_add+'time.txt','a+')
f.write('time for dynamic programming: '+ str(dynamicTime)+'\n' )
#f.write(str(t))
#f.write('\n')
for row in t_list:
    f.write(str(row)+'\n')
f.close()

