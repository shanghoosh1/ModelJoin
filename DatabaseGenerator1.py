import numpy as np
import csv
f= open('synthetic_data/DB1/db_discription.txt','w')
tbl_num=13
tbl_sizes=[10000000,10000000,10000000,10000000,10000000,10000000,10000000,10000000,10000000,10000000,10000000,10000000,10000000]
joinAtt_NDVs=[1000000,1000000,1000000,1000000,1000000,1000000,1000000,500000,100000,50000,10000,5000]
#1 percentage of first att,#2 percentage of second att overall  ==> base DVs for second att,#3 minimum percentage of values in base DVs,#4 maximum percentage of values  in base DVs
conditional_NDVs_percents=[[0.8,0.8,0.05,0.1],[0.7,0.7,0.05,0.1],[0.6,0.6,0.05,0.1],[0.5,0.5,0.2,0.25],[0.5,0.5,0.15,0.2],[0.5,0.5,0.1,0.15],[0.5,0.5,0.05,0.1],[0.5,0.5,0.05,0.1],[0.5,0.5,0.05,0.1],[0.5,0.5,0.05,0.1],[0.5,0.5,0.05,0.1],]

first_lastPercent=0.5
st='table sizes: '
st += ", ".join(str(x) for x in tbl_sizes)
f.write(st+'\n')
st='maximum NDVs for each join att: '
st += ", ".join(str(x) for x in joinAtt_NDVs)
f.write(st+'\n')
st='conditional percentages: '
st += ", ".join(str(x) for x in conditional_NDVs_percents)
f.write(st+'\n')
f.write('first and last join att percentages: '+ str(first_lastPercent)+'\n')
DVs={}
for i,ndv in enumerate(joinAtt_NDVs):
    DVs.update({i:['dv_'+str(i)+'_'+ str(j) for j in range(ndv)]})
import time
for i in range(tbl_num):
    t=time.time()
    att1=[]
    att2=[]
    if i==0:
        att1 = np.random.normal(0, 1, tbl_sizes[i])
        dv2 = np.random.choice((DVs[i]), int(first_lastPercent*joinAtt_NDVs[i]))
        att2 = np.random.choice(dv2, tbl_sizes[i])
        att1 = np.array(att1).reshape(len(att1), 1)
        att2 = np.array(att2).reshape(len(att2), 1)
        tbl = np.append(att1, att2, axis=1)
        np.savetxt('synthetic_data/DB1/tbl_' + str(i) + '.csv', tbl.astype('str'), delimiter=",", fmt='%s %s')
        print('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(len(set(att1[:, 0]))) + ', ' + str(
            len(set(att2[:, 0]))) + ' with time: ' + str(time.time() - t) + '\n')
        f.write('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(len(set(att1[:, 0]))) + ', ' + str(
            len(set(att2[:, 0]))) + ' with time: ' + str(time.time() - t) + '\n')
        att1=None
        att2=None
        tbl=None
    elif i==tbl_num-1:
        dv1 = np.random.choice((DVs[i-1]), int(first_lastPercent*joinAtt_NDVs[i-1]))
        att1 = np.random.choice(dv1, tbl_sizes[i])
        att2 = np.random.normal(0, 1, tbl_sizes[i])
        att1 = np.array(att1).reshape(len(att1), 1)
        att2 = np.array(att2).reshape(len(att2), 1)
        tbl = np.append(att1, att2, axis=1)
        np.savetxt('synthetic_data/DB1/tbl_' + str(i) + '.csv', tbl.astype('str'), delimiter=",", fmt='%s %s')
        print('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(len(set(att1[:, 0]))) + ', ' + str(
            len(set(att2[:, 0]))) + ' with time: ' + str(time.time() - t) + '\n')
        f.write('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(len(set(att1[:, 0]))) + ', ' + str(
            len(set(att2[:, 0]))) + ' with time: ' + str(time.time() - t) + '\n')
        att1=None
        att2=None
        tbl=None
    else:
        att_ndv1 = conditional_NDVs_percents[i - 1]
        dv1 = np.random.choice((DVs[i - 1]), int(att_ndv1[0] * joinAtt_NDVs[i - 1]))
        att1 = np.random.choice(dv1, tbl_sizes[i])
        att1_disc=np.array(np.unique(att1, return_counts=True))
        att_ndv = conditional_NDVs_percents[i - 1]
        dv_base = np.random.choice((DVs[i]), int(att_ndv[1] * joinAtt_NDVs[i]))
        tbl=[]
        ss=0
        for ind,v in enumerate(att1_disc[0]):
            conditional_DVs=np.random.choice(dv_base, int(np.random.uniform(att_ndv[2], att_ndv[3]) * len(dv_base)))
            att2=list(np.random.choice(conditional_DVs, int(att1_disc[1][ind])))
            v=[v]*len(att2)
            att2 = np.array(att2).reshape(len(att2), 1)
            v = np.array(v).reshape(len(v), 1)
            tbl += list(np.append(v, att2, axis=1))
            ss=len(tbl)
            print(str(ss)+' data generated for table: '+str(i))
        tbl=np.array(tbl)
        np.savetxt('synthetic_data/DB1/tbl_' + str(i) + '.csv', tbl.astype('str'), delimiter=",", fmt='%s %s')

        xyNDVs = [tuple(a) for a in tbl]
        xyNDVs = len(set(xyNDVs))
        att2_nvd=len(set(tbl[:, 1]))
        att1_nvd=len(set(tbl[:, 0]))
        allNvds=att1_nvd*att2_nvd
        ratio1=xyNDVs/allNvds
        ratio2=xyNDVs/tbl_sizes[i]

        print('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(att1_nvd) + ', ' + str(
            att2_nvd) + ' x_nvd*y_nvd: ' + str(allNvds) + ', xy_NDVs: ' + str(xyNDVs) + ', xy_NDVs/x_nvd*y_nvd: ' + str(
            ratio1) + ', xy_NDVs/table_size: ' + str(ratio2) + ' with time: ' + str(time.time() - t) + '\n')

        f.write('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(att1_nvd) + ', ' + str(
            att2_nvd)+ ' x_nvd*y_nvd: '+ str(allNvds) +', xy_NDVs: '+str(xyNDVs)+', xy_NDVs/x_nvd*y_nvd: '+str(ratio1) +', xy_NDVs/table_size: '+str(ratio2)+ ' with time: ' + str(time.time() - t) + '\n')
        att1=None
        att2=None
        tbl=None
f.close()
#
# f= open('synthetic_data/db_discription2.txt','w')
# tbl_num=8
# tbl_sizes=[100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000]
# joinAtt_NDVs_2=[100000,100000,100000,100000,100000,100000,100000,100000]
# conditional_NDVs_percents_2=[[1,1,0.0005,0.0001],[1,1,0.0001,0.005],[1,1,0.005,0.01],[1,1,0.01,0.15],[1,1,0.15,0.2],[1,1,0.2,0.25],[1,1,0.25,0.3],[1,1,0.3,0.35]]
#
# first_lastPercent=1
# st='table sizes: '
# st += ", ".join(str(x) for x in tbl_sizes)
# f.write(st+'\n')
# st='maximum NDVs for each join att: '
# st += ", ".join(str(x) for x in joinAtt_NDVs_2)
# f.write(st+'\n')
# st='conditional percentages: '
# st += ", ".join(str(x) for x in conditional_NDVs_percents_2)
# f.write(st+'\n')
# f.write('first and last join att percentages: '+ str(first_lastPercent)+'\n')
# DVs={}
# for i,ndv in enumerate(joinAtt_NDVs_2):
#     DVs.update({i:['dv_'+str(i)+'_'+ str(j) for j in range(ndv)]})
# import time
# for i in range(tbl_num):
#     t=time.time()
#     att1=[]
#     att2=[]
#     att_ndv1 = conditional_NDVs_percents_2[i]
#     dv1 = np.random.choice((DVs[i]), int(att_ndv1[0] * joinAtt_NDVs_2[i]))
#     att1 = np.random.choice(dv1, tbl_sizes[i])
#     att1_disc=np.array(np.unique(att1, return_counts=True))
#     dv_base = np.random.choice((DVs[i+1]), int(att_ndv1[1] * joinAtt_NDVs_2[i+1]))
#     tbl=[]
#     ss=0
#     for ind,v in enumerate(att1_disc[0]):
#         conditional_DVs=np.random.choice(dv_base, int(np.random.uniform(att_ndv1[2], att_ndv1[3]) * len(dv_base)))
#         att2=list(np.random.choice(conditional_DVs, int(att1_disc[1][ind])))
#         v=[v]*len(att2)
#         att2 = np.array(att2).reshape(len(att2), 1)
#         v = np.array(v).reshape(len(v), 1)
#         tbl += list(np.append(v, att2, axis=1))
#         ss=len(tbl)
#         print(str(ss)+' data generated for table: '+str(i))
#     tbl=np.array(tbl)
#     np.savetxt('synthetic_data/tbl2_' + str(i) + '.csv', tbl.astype('str'), delimiter=",", fmt='%s %s')
#
#     xyNDVs = [tuple(a) for a in tbl]
#     xyNDVs = len(set(xyNDVs))
#     att2_nvd=len(set(tbl[:, 1]))
#     att1_nvd=len(set(tbl[:, 0]))
#     allNvds=att1_nvd*att2_nvd
#     ratio1=xyNDVs/allNvds
#     ratio2=xyNDVs/tbl_sizes[i]
#
#     print('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(att1_nvd) + ', ' + str(
#         att2_nvd) + ' x_nvd*y_nvd: ' + str(allNvds) + ', xy_NDVs: ' + str(xyNDVs) + ', xy_NDVs/x_nvd*y_nvd: ' + str(
#         ratio1) + ', xy_NDVs/table_size: ' + str(ratio2) + ' with time: ' + str(time.time() - t) + '\n')
#
#     f.write('Generated NDVs for table ' + str(i) + '(att1,att2):  ' + str(att1_nvd) + ', ' + str(
#         att2_nvd)+ ' x_nvd*y_nvd: '+ str(allNvds) +', xy_NDVs: '+str(xyNDVs)+', xy_NDVs/x_nvd*y_nvd: '+str(ratio1) +', xy_NDVs/table_size: '+str(ratio2)+ ' with time: ' + str(time.time() - t) + '\n')
#     att1=None
#     att2=None
#     tbl=None
# f.close()

