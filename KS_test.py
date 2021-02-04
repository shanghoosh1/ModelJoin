import json
import pandas as pd
import numpy as np
# import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
# out="synthetic_data/DB2_small/3join/"
# data1 = pd.read_csv("synthetic_data/DB2_small/3join/exactjoin3.csv",delimiter=',', usecols=[1,2])
# data1.columns=['a','b']
# data1=data1.sort_values(by=['a','b'])
# data=data1.values
# data=[ str(int(a[0]))+'_'+str(int(a[1]))for a in data]
# data=np.array(data)
# ss=set(data)
# samples1 = pd.read_csv("synthetic_data/DB2_small/3join/sample.csv",delimiter=',')
# samples1.columns=['a','b']
# samples1=samples1.sort_values(by=['a','b'])
# samples=samples1.values
# samples=[ str(int(a[0]))+'_'+str(int(a[1])) for a in samples]
# # samples=np.array()
# # samples=samples[0:2000,:]
# s=set(samples)
#
# result = pd.concat([data1,samples1])
# result=result.sort_values(by=['a','b'])
# result=result.values
# result=[ str(int(a[0]))+'_'+str(int(a[1])) for a in result]
# result=np.array(result)
# sss=set(result)
#
# dic_mqin={}
# index=0
# for i in result:
#     if i not in dic_mqin:
#         dic_mqin.update({i:index})
#         index+=1
#
# print('df')
# import time
# FP = {}
# TN = {}
# TP = {}
# FN = {}
# t=time.time()
# unique_elements, counts_elements = np.unique(data, return_counts=True)
# unique_elementsS, counts_elementsS = np.unique(samples, return_counts=True)
# realy_dict={}
# fakey_dict={}
# sum1=sum(counts_elements)
# sum2=sum(counts_elementsS)
# for k,v in zip(unique_elements,counts_elements):
#     realy_dict.update({k:v/sum1})
# for k,v in zip(unique_elementsS,counts_elementsS):
#     fakey_dict.update({k:v/sum2})
#
# dt=[dic_mqin[i] for i in data]
# sm=[dic_mqin[i] for i in samples]
# from scipy import stats
# aaa=stats.ks_2samp(dt, sm)
# print('KS: '+ str(aaa))
# pltList1=[]
# pltList2=[]
# sss=0
# sss2=0
# for i in dic_mqin:
#     if i in realy_dict:
#         sss=sss+realy_dict[i]
#         pltList1.append([dic_mqin[i],sss])
#     else:
#         pltList1.append([dic_mqin[i], sss])
#
#     if i in fakey_dict:
#         sss2 = sss2 + fakey_dict[i]
#         pltList2.append([dic_mqin[i], sss2])
#     else:
#         pltList2.append([dic_mqin[i], sss2])
# import matplotlib.pyplot as plt
# pltList1=np.array(pltList1)
# pltList2=np.array(pltList2)
# import math
# a11=math.sqrt(-0.5*math.log(0.01/2))*math.sqrt((len(data)+20000)/(len(data)*20000))
# pltList11=[]
# pltList12=[]
# for i in pltList1:
#     pltList11.append([i[0],i[1]+a11])
#     pltList12.append([i[0],i[1]-a11])
# pltList11=np.array(pltList11)
# pltList12=np.array(pltList12)
# plt.plot(pltList1[:,0], pltList1[:,1],'r',label='Exact CDF');
# plt.plot(pltList1[:,0], pltList11[:,1],'y',label='Boundaries');
# plt.plot(pltList1[:,0], pltList12[:,1],'y');
# plt.plot(pltList2[:,0],pltList2[:,1],'b', label='Approximate CDF');
# plt.xlabel('Tuple ID', fontsize=10)
# plt.ylabel('CDF', fontsize=10)
# plt.legend()
# plt.savefig(out+'1KS.png')
# # plt.show()
# print(sum(fakey_dict.values()))
# print(sum(realy_dict.values()))
# tt=time.time()
# for key in fakey_dict.keys():
#     if key in realy_dict:
#         if fakey_dict[key] >= realy_dict[key]:
#             if key in TP:
#                 TP[key]=TP[key]+realy_dict[key]
#             else:
#                 TP.update({key: realy_dict[key]})
#
#             if key in TN:
#                 TN[key] = TN[key] + 1 - fakey_dict[key]
#             else:
#                 TN.update({key: 1 - fakey_dict[key]})
#
#             if key in FP:
#                 FP[key] = FP[key] + fakey_dict[key] - realy_dict[key]
#             else:
#                 FP.update({key: fakey_dict[key] - realy_dict[key]})
#
#             if key in FN:
#                 FN[key] = FN[key] + 0
#             else:
#                 FN.update({key: 0})
#
#         else:
#             if key in TP:
#                 TP[key] = TP[key] + fakey_dict[key]
#             else:
#                 TP.update({key: fakey_dict[key]})
#
#             if key in TN:
#                 TN[key] = TN[key] +  1-realy_dict[key]
#             else:
#                 TN.update({key: 1-realy_dict[key]})
#
#             if key in FP:
#                 FP[key] = FP[key] + 0
#             else:
#                 FP.update({key: 0})
#
#             if key in FN:
#                 FN[key] = FN[key] + realy_dict[key] - fakey_dict[key]
#             else:
#                 FN.update({key: realy_dict[key] - fakey_dict[key]})
#
#     else:
#         if key in TP:
#             TP[key] = TP[key] + 0
#         else:
#             TP.update({key: 0})
#
#         if key in TN:
#             TN[key] = TN[key] + 1 - fakey_dict[key]
#         else:
#             TN.update({key: 1 - fakey_dict[key]})
#
#         if key in FP:
#             FP[key] = FP[key] + fakey_dict[key]
#         else:
#             FP.update({key: fakey_dict[key]})
#
#         if key in FN:
#             FN[key] = FN[key] + 0
#         else:
#             FN.update({key: 0})
# print(sum(FP.values()))
# print(sum(FN.values()))
# Precision = sum(TP.values()) /( sum(TP.values()) + sum(FP.values()))
# Recall = sum(TP.values()) / (sum(TP.values()) + sum(FN.values()))
# F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
# print('Precision: '+str(Precision))
# print('Recall: '+str(Recall))
# print('F_score: '+str(F1_Score))
#
#
# FNN=0
# FPP=0
# for k in dic_mqin.keys():
#     if k in realy_dict and k in fakey_dict:
#         if realy_dict[k]>fakey_dict[k]:
#             FNN+=realy_dict[k]-fakey_dict[k]
#         else:
#             FPP+=fakey_dict[k]-realy_dict[k]
#     elif not (k in realy_dict) and (k in fakey_dict):
#         FPP += fakey_dict[k]
#     elif  (k in realy_dict) and not(k in fakey_dict):
#         FNN += realy_dict[k]
# print(FNN)
# print(FPP)
# Precision = sum(TP.values()) /( sum(TP.values()) + FPP)
# Recall = sum(TP.values()) / (sum(TP.values()) + FNN)
# F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
# print('Precision: '+str(Precision))
# print('Recall: '+str(Recall))
# print('F_score: '+str(F1_Score))
# f1=open(out+'result.txt', 'a+')
# f1.write('Precision: '+str(Precision))
# f1.write('\n')
# f1.write(str(aaa))
# f1.close()

################################################################################################################
# import json
# import pandas as pd
# import numpy as np
# # import pandas as pd
# import psycopg2
# import matplotlib.pyplot as plt
# out="synthetic_data/tpc_ds/nips/join/"
# data1 = pd.read_csv("synthetic_data/tpc_ds/nips/join/join4.csv",delimiter=',', usecols=[0,1,2,3,4,5])
# data1.columns=['a','b','c','d','e','f']
# data1=data1.sort_values(by=['a','b','c','d','e','f'])
# data=data1.values
# data=[ str(int(a[0]))+'_'+str(int(a[1])) +'_'+ str(int(a[2]))+'_'+str(int(a[3])) +'_'+ str(int(a[4]))+'_'+str(int(a[5])) for a in data]
# data=np.array(data)
# ss=set(data)
# samples1 = pd.read_csv("synthetic_data/tpc_ds/nips/join/sample.csv",delimiter=',')
# samples1.columns=['a','b','c','d','e','f']
# samples1=samples1.sort_values(by=['a','b','c','d','e','f'])
# samples=samples1.values
# samples=[ str(int(a[0]))+'_'+str(int(a[1]))+'_'+ str(int(a[2]))+'_'+str(int(a[3]))+'_'+ str(int(a[4]))+'_'+str(int(a[5])) for a in samples]
# # samples=np.array()
# # samples=samples[0:2000,:]
# s=set(samples)
#
# result = pd.concat([data1,samples1])
# result=result.sort_values(by=['a','b','c','d','e','f'])
# result=result.values
# result=[ str(int(a[0]))+'_'+str(int(a[1]))+'_'+ str(int(a[2]))+'_'+str(int(a[3]))+'_'+ str(int(a[4]))+'_'+str(int(a[5])) for a in result]
# result=np.array(result)
# sss=set(result)
#
# dic_mqin={}
# index=0
# for i in result:
#     if i not in dic_mqin:
#         dic_mqin.update({i:index})
#         index+=1
#
# print('df')
# import time
# FP = {}
# TN = {}
# TP = {}
# FN = {}
# t=time.time()
# unique_elements, counts_elements = np.unique(data, return_counts=True)
# unique_elementsS, counts_elementsS = np.unique(samples, return_counts=True)
# realy_dict={}
# fakey_dict={}
# sum1=sum(counts_elements)
# sum2=sum(counts_elementsS)
# for k,v in zip(unique_elements,counts_elements):
#     realy_dict.update({k:v/sum1})
# for k,v in zip(unique_elementsS,counts_elementsS):
#     fakey_dict.update({k:v/sum2})
#
# dt=[dic_mqin[i] for i in data]
# sm=[dic_mqin[i] for i in samples]
# from scipy import stats
# aaa=stats.ks_2samp(dt, sm)
# print('KS: '+ str(aaa))
# pltList1=[]
# pltList2=[]
# sss=0
# sss2=0
# for i in dic_mqin:
#     if i in realy_dict:
#         sss=sss+realy_dict[i]
#         pltList1.append([dic_mqin[i],sss])
#     else:
#         pltList1.append([dic_mqin[i], sss])
#
#     if i in fakey_dict:
#         sss2 = sss2 + fakey_dict[i]
#         pltList2.append([dic_mqin[i], sss2])
#     else:
#         pltList2.append([dic_mqin[i], sss2])
# import matplotlib.pyplot as plt
# pltList1=np.array(pltList1)
# pltList2=np.array(pltList2)
# import math
# a01=math.sqrt(-0.5*math.log(0.01/2))*math.sqrt((len(data)+500000)/(len(data)*500000))
# print("borders alpha 0.01================="+ str(a01))
#
# pltList11=[]
# pltList12=[]
# for i in pltList1:
#     pltList11.append([i[0],i[1]+a01])
#     pltList12.append([i[0],i[1]-a01])
# pltList11=np.array(pltList11)
# pltList12=np.array(pltList12)
#
# a001=math.sqrt(-0.5*math.log(0.0001/2))*math.sqrt((len(data)+500000)/(len(data)*500000))
# print("borders alpha 0.001================="+ str(a001))
#
# pltList13=[]
# pltList14=[]
# for i in pltList1:
#     pltList13.append([i[0],i[1]+a001])
#     pltList14.append([i[0],i[1]-a001])
# pltList13=np.array(pltList13)
# pltList14=np.array(pltList14)
# plt.plot(pltList1[:,0], pltList13[:,1],'g',label='Boundaries alpha=0.0001');
# plt.plot(pltList1[:,0], pltList14[:,1],'g');
#
# plt.plot(pltList1[:,0], pltList1[:,1],'r',label='Exact CDF');
# plt.plot(pltList1[:,0], pltList11[:,1],'y',label='Boundaries alpha=0.01');
# plt.plot(pltList1[:,0], pltList12[:,1],'y');
# plt.plot(pltList2[:,0],pltList2[:,1],'b', label='Approximate CDF');
# plt.xlabel('Tuple ID', fontsize=10)
# plt.ylabel('CDF', fontsize=10)
# plt.legend()
# plt.savefig(out+'KS.png')
# plt.show()
# print(sum(fakey_dict.values()))
# print(sum(realy_dict.values()))
# tt=time.time()
# for key in fakey_dict.keys():
#     if key in realy_dict:
#         if fakey_dict[key] >= realy_dict[key]:
#             if key in TP:
#                 TP[key]=TP[key]+realy_dict[key]
#             else:
#                 TP.update({key: realy_dict[key]})
#
#             if key in TN:
#                 TN[key] = TN[key] + 1 - fakey_dict[key]
#             else:
#                 TN.update({key: 1 - fakey_dict[key]})
#
#             if key in FP:
#                 FP[key] = FP[key] + fakey_dict[key] - realy_dict[key]
#             else:
#                 FP.update({key: fakey_dict[key] - realy_dict[key]})
#
#             if key in FN:
#                 FN[key] = FN[key] + 0
#             else:
#                 FN.update({key: 0})
#
#         else:
#             if key in TP:
#                 TP[key] = TP[key] + fakey_dict[key]
#             else:
#                 TP.update({key: fakey_dict[key]})
#
#             if key in TN:
#                 TN[key] = TN[key] +  1-realy_dict[key]
#             else:
#                 TN.update({key: 1-realy_dict[key]})
#
#             if key in FP:
#                 FP[key] = FP[key] + 0
#             else:
#                 FP.update({key: 0})
#
#             if key in FN:
#                 FN[key] = FN[key] + realy_dict[key] - fakey_dict[key]
#             else:
#                 FN.update({key: realy_dict[key] - fakey_dict[key]})
#
#     else:
#         if key in TP:
#             TP[key] = TP[key] + 0
#         else:
#             TP.update({key: 0})
#
#         if key in TN:
#             TN[key] = TN[key] + 1 - fakey_dict[key]
#         else:
#             TN.update({key: 1 - fakey_dict[key]})
#
#         if key in FP:
#             FP[key] = FP[key] + fakey_dict[key]
#         else:
#             FP.update({key: fakey_dict[key]})
#
#         if key in FN:
#             FN[key] = FN[key] + 0
#         else:
#             FN.update({key: 0})
# # print(sum(FP.values()))
# # print(sum(FN.values()))
# Precision = sum(TP.values()) /( sum(TP.values()) + sum(FP.values()))
# Recall = sum(TP.values()) / (sum(TP.values()) + sum(FN.values()))
# F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
# # print('Precision: '+str(Precision))
# # print('Recall: '+str(Recall))
# # print('F_score: '+str(F1_Score))
#
#
# FNN=0
# FPP=0
# for k in dic_mqin.keys():
#     if k in realy_dict and k in fakey_dict:
#         if realy_dict[k]>fakey_dict[k]:
#             FNN+=realy_dict[k]-fakey_dict[k]
#         else:
#             FPP+=fakey_dict[k]-realy_dict[k]
#     elif not (k in realy_dict) and (k in fakey_dict):
#         FPP += fakey_dict[k]
#     elif  (k in realy_dict) and not(k in fakey_dict):
#         FNN += realy_dict[k]
# print(FNN)
# print(FPP)
# Precision = sum(TP.values()) /( sum(TP.values()) + FPP)
# Recall = sum(TP.values()) / (sum(TP.values()) + FNN)
# F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
# print('Precision: '+str(Precision))
# print('Recall: '+str(Recall))
# print('F_score: '+str(F1_Score))
# f1=open(out+'result.txt', 'a+')
# f1.write('Precision: '+str(Precision))
# f1.write('\n')
# f1.write(str(aaa))
# f1.write('\n')
# f1.write('alpha 0.01= '+str(a01)+',    alpha 0.001= ' +str(a001))
# f1.close()
#############################################################################################################
import json
import pandas as pd
import numpy as np
# import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
out="synthetic_data/tpc_ds/nips/2T/join/"
data1 = pd.read_csv("synthetic_data/tpc_ds/nips/2T/join/join4.csv",delimiter=',', usecols=[0,3,5])
data1.columns=['a','b','c']
data1=data1.sort_values(by=['a','b','c'])
data=data1.values
data=[ str(int(a[0]))+'_'+str(int(a[1])) +'_'+str(int(a[2])) for a in data]
data=np.array(data)
ss=set(data)
samples1 = pd.read_csv("synthetic_data/tpc_ds/nips/2T/join/sample.csv",delimiter=',', usecols=[0,3,5])
samples1.columns=['a','b','c']
samples1=samples1.sort_values(by=['a','b','c'])
samples=samples1.values
samples=[ str(int(a[0]))+'_'+str(int(a[1]))+'_'+str(int(a[2])) for a in samples]
s=set(samples)

samples2 = pd.read_csv("synthetic_data/tpc_ds/nips/2T/join/join4_60samples.csv",delimiter=',', usecols=[0,3,5])
samples2.columns=['a','b','c']
samples2=samples2.sort_values(by=['a','b','c'])
samplesNew=samples2.values
samplesNew=[ str(int(a[0]))+'_'+str(int(a[1]))+'_'+str(int(a[2])) for a in samplesNew]
sNew=set(samplesNew)

result = pd.concat([data1,samples1])
result = pd.concat([result,samples2])
result=result.sort_values(by=['a','b','c'])
result=result.values
result=[ str(int(a[0]))+'_'+str(int(a[1])) +'_'+str(int(a[2])) for a in result]
result=np.array(result)
sss=set(result)
samples1_size=len(samples1)
samples2_size=len(samples2)
dic_mqin={}
index=0
for i in result:
    if i not in dic_mqin:
        dic_mqin.update({i:index})
        index+=1

print('len data:'+ str(len(data)))
import time
FP = {}
TN = {}
TP = {}
FN = {}
t=time.time()
unique_elements, counts_elements = np.unique(data, return_counts=True)
unique_elementsS, counts_elementsS = np.unique(samples, return_counts=True)
unique_elements2, counts_elements2 = np.unique(samplesNew, return_counts=True)
realy_dict={}
fakey_dict={}
unif_dict={}
sum1=sum(counts_elements)
sum2=sum(counts_elementsS)
sum3=sum(counts_elements2)
for k,v in zip(unique_elements,counts_elements):
    realy_dict.update({k:v/sum1})
for k,v in zip(unique_elementsS,counts_elementsS):
    fakey_dict.update({k:v/sum2})
for k,v in zip(unique_elements2,counts_elements2):
    unif_dict.update({k:v/sum3})

dt=[dic_mqin[i] for i in data]
sm=[dic_mqin[i] for i in samples]
sm2=[dic_mqin[i] for i in samplesNew]
from scipy import stats
aaa=stats.ks_2samp(dt, sm)
print('KS1: '+ str(aaa))
aaa2=stats.ks_2samp(dt, sm2)
print('KS2: '+ str(aaa2))
pltList1=[]
pltList2=[]
pltListNew=[]
sss=0
sss2=0
sss3=0
for i in dic_mqin:
    if i in realy_dict:
        sss=sss+realy_dict[i]
        pltList1.append([dic_mqin[i],sss])
    else:
        pltList1.append([dic_mqin[i], sss])
    if i in fakey_dict:
        sss2 = sss2 + fakey_dict[i]
        pltList2.append([dic_mqin[i], sss2])
    else:
        pltList2.append([dic_mqin[i], sss2])
    if i in unif_dict:
        sss3 = sss3 + unif_dict[i]
        pltListNew.append([dic_mqin[i], sss3])
    else:
        pltListNew.append([dic_mqin[i], sss3])
import matplotlib.pyplot as plt
pltList1=np.array(pltList1)
pltList2=np.array(pltList2)
pltListNew=np.array(pltListNew)
import math

a01=math.sqrt(-0.5*math.log(0.01/2))*math.sqrt((len(data)+samples1_size)/(len(data)*samples1_size))
print("borders alpha 0.01================="+ str(a01))
a01_2=math.sqrt(-0.5*math.log(0.01/2))*math.sqrt((len(data)+samples2_size)/(len(data)*samples2_size))
print("borders alpha 0.01 for second================="+ str(a01_2))

pltList11=[]
pltList12=[]
for i in pltList1:
    pltList11.append([i[0],i[1]+a01])
    pltList12.append([i[0],i[1]-a01])
pltList11=np.array(pltList11)
pltList12=np.array(pltList12)

pltListNew1=[]
pltListNew2=[]
for i in pltList1:
    pltListNew1.append([i[0],i[1]+a01_2])
    pltListNew2.append([i[0],i[1]-a01_2])
pltListNew1=np.array(pltListNew1)
pltListNew2=np.array(pltListNew2)
# a001=math.sqrt(-0.5*math.log(0.001/2))*math.sqrt((len(data)+2000)/(len(data)*2000))
# print("borders alpha 0.0001================="+ str(a001))
#
# pltList13=[]
# pltList14=[]
# for i in pltList1:
#     pltList13.append([i[0],i[1]+a001])
#     pltList14.append([i[0],i[1]-a001])
# pltList13=np.array(pltList13)
# pltList14=np.array(pltList14)
# plt.plot(pltList1[:,0], pltList13[:,1],'g',label='Boundaries alpha=0.001');
# plt.plot(pltList1[:,0], pltList14[:,1],'g');

plt.plot(pltList1[:,0], pltList1[:,1],'r',label='Exact CDF',linewidth=0.8);
plt.plot(pltList1[:,0], pltList11[:,1],'y',label='Uni-Boundaries',linewidth=0.8);
plt.plot(pltList1[:,0], pltList12[:,1],'y',linewidth=0.8);
plt.plot(pltList2[:,0],pltList2[:,1],'b', label='Uni-Approximate CDF',linewidth=0.8);
plt.plot(pltListNew1[:,0], pltList11[:,1],'g',label='10% sample-join boundaries',linewidth=0.8);
plt.plot(pltListNew2[:,0], pltList12[:,1],'g',linewidth=0.8);
plt.plot(pltListNew[:,0],pltListNew[:,1],'black', label='10% Sample-Join CDF',linewidth=0.8);
plt.xlabel('Tuple ID        Alpha=0.01', fontsize=10)
plt.ylabel('CDFs and Boundaries', fontsize=10)
plt.legend()
plt.savefig(out+'KS.png')
plt.show()
f1=open(out+'result.txt', 'a+')
f1.write('ks for uniSample'+str(aaa))
f1.write('\n')
f1.write('ks for SampleJoin'+str(aaa))
f1.write('\n')
f1.write('alpha 0.01= '+str(a01)+',    alpha 0.01 for sampleJoin= ' +str(a01_2))
f1.close()

##################################################################import json
import pandas as pd
import numpy as np
# import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
out="synthetic_data/tpc_ds/nips/3T/join/"
data1 = pd.read_csv("synthetic_data/tpc_ds/nips/3T/join/join3t.csv",delimiter=',', usecols=[0,2,4,6])
data1.columns=['a','b','c','d']
data1=data1.sort_values(by=['a','b','c','d'])
data=data1.values
data=[ str(int(a[0]))+'_'+str(int(a[1])) +'_'+str(int(a[2]))+'_'+str(int(a[3])) for a in data]
data=np.array(data)
ss=set(data)
samples1 = pd.read_csv("synthetic_data/tpc_ds/nips/3T/join/sample.csv",delimiter=',', usecols=[0,2,4,6])

samples1.columns=['a','b','c','d']
samples1=samples1.sort_values(by=['a','b','c','d'])
samples=samples1.values
samples=[ str(int(a[0]))+'_'+str(int(a[1]))+'_'+str(int(a[2]))+'_'+str(int(a[3])) for a in samples]
s=set(samples)

samples2 = pd.read_csv("synthetic_data/tpc_ds/nips/3T/join/join33t.csv",delimiter=',', usecols=[0,2,4,6])
samples2.columns=['a','b','c','d']
samples2=samples2.sort_values(by=['a','b','c','d'])
samplesNew=samples2.values
samplesNew=[ str(int(a[0]))+'_'+str(int(a[1]))+'_'+str(int(a[2]))+'_'+str(int(a[3])) for a in samplesNew]
sNew=set(samplesNew)
samples1_size=len(samples1)
samples2_size=len(samples2)
result = pd.concat([data1,samples1])
result = pd.concat([result,samples2])
result=result.sort_values(by=['a','b','c','d'])
result=result.values
result=[ str(int(a[0]))+'_'+str(int(a[1])) +'_'+str(int(a[2]))+'_'+str(int(a[3])) for a in result]
result=np.array(result)
sss=set(result)

dic_mqin={}
index=0
for i in result:
    if i not in dic_mqin:
        dic_mqin.update({i:index})
        index+=1

print('len data:'+ str(len(data)))
import time
FP = {}
TN = {}
TP = {}
FN = {}
t=time.time()
unique_elements, counts_elements = np.unique(data, return_counts=True)
unique_elementsS, counts_elementsS = np.unique(samples, return_counts=True)
unique_elements2, counts_elements2 = np.unique(samplesNew, return_counts=True)
realy_dict={}
fakey_dict={}
unif_dict={}
sum1=sum(counts_elements)
sum2=sum(counts_elementsS)
sum3=sum(counts_elements2)
for k,v in zip(unique_elements,counts_elements):
    realy_dict.update({k:v/sum1})
for k,v in zip(unique_elementsS,counts_elementsS):
    fakey_dict.update({k:v/sum2})
for k,v in zip(unique_elements2,counts_elements2):
    unif_dict.update({k:v/sum3})

dt=[dic_mqin[i] for i in data]
sm=[dic_mqin[i] for i in samples]
sm2=[dic_mqin[i] for i in samplesNew]
from scipy import stats
aaa=stats.ks_2samp(dt, sm)
print('KS1: '+ str(aaa))
aaa2=stats.ks_2samp(dt, sm2)
print('KS2: '+ str(aaa2))
pltList1=[]
pltList2=[]
pltListNew=[]
sss=0
sss2=0
sss3=0
for i in dic_mqin:
    if i in realy_dict:
        sss=sss+realy_dict[i]
        pltList1.append([dic_mqin[i],sss])
    else:
        pltList1.append([dic_mqin[i], sss])
    if i in fakey_dict:
        sss2 = sss2 + fakey_dict[i]
        pltList2.append([dic_mqin[i], sss2])
    else:
        pltList2.append([dic_mqin[i], sss2])
    if i in unif_dict:
        sss3 = sss3 + unif_dict[i]
        pltListNew.append([dic_mqin[i], sss3])
    else:
        pltListNew.append([dic_mqin[i], sss3])
import matplotlib.pyplot as plt
pltList1=np.array(pltList1)
pltList2=np.array(pltList2)
pltListNew=np.array(pltListNew)
import math

a01=math.sqrt(-0.5*math.log(0.01/2))*math.sqrt((len(data)+samples1_size)/(len(data)*samples1_size))
print("borders alpha 0.01================="+ str(a01))
a01_2=math.sqrt(-0.5*math.log(0.01/2))*math.sqrt((len(data)+samples2_size)/(len(data)*samples2_size))
print("borders alpha 0.01 for the second================="+ str(a01_2))

pltList11=[]
pltList12=[]
for i in pltList1:
    pltList11.append([i[0],i[1]+a01])
    pltList12.append([i[0],i[1]-a01])
pltList11=np.array(pltList11)
pltList12=np.array(pltList12)

pltListNew1=[]
pltListNew2=[]
for i in pltList1:
    pltListNew1.append([i[0],i[1]+a01_2])
    pltListNew2.append([i[0],i[1]-a01_2])
pltListNew1=np.array(pltListNew1)
pltListNew2=np.array(pltListNew2)
# a001=math.sqrt(-0.5*math.log(0.001/2))*math.sqrt((len(data)+2000)/(len(data)*2000))
# print("borders alpha 0.0001================="+ str(a001))
#
# pltList13=[]
# pltList14=[]
# for i in pltList1:
#     pltList13.append([i[0],i[1]+a001])
#     pltList14.append([i[0],i[1]-a001])
# pltList13=np.array(pltList13)
# pltList14=np.array(pltList14)
# plt.plot(pltList1[:,0], pltList13[:,1],'g',label='Boundaries alpha=0.001');
# plt.plot(pltList1[:,0], pltList14[:,1],'g');

plt.plot(pltList1[:,0], pltList1[:,1],'r',label='Exact CDF',linewidth=0.8);
plt.plot(pltList1[:,0], pltList11[:,1],'y',label='Uni-Boundaries',linewidth=0.8);
plt.plot(pltList1[:,0], pltList12[:,1],'y',linewidth=0.8);
plt.plot(pltList2[:,0],pltList2[:,1],'b', label='Uni-Approximate CDF',linewidth=0.8);
plt.plot(pltListNew1[:,0], pltList11[:,1],'g',label='10% sample-join boundaries',linewidth=0.8);
plt.plot(pltListNew2[:,0], pltList12[:,1],'g',linewidth=0.8);
plt.plot(pltListNew[:,0],pltListNew[:,1],'black', label='10% Sample-Join CDF',linewidth=0.8);
plt.xlabel('Tuple ID         alpha=0.01', fontsize=10)
plt.ylabel('CDFs and Boundaries', fontsize=10)
plt.legend()
plt.savefig(out+'KS.png')
plt.show()
f1=open(out+'result.txt', 'a+')
f1.write('ks for uniSample'+str(aaa))
f1.write('\n')
f1.write('ks for SampleJoin'+str(aaa))
f1.write('\n')
f1.write('alpha 0.01= '+str(a01)+',    alpha 0.01 for sampleJoin= ' +str(a01_2))
f1.close()