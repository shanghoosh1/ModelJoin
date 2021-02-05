import numpy as np
import time
# f= open('synthetic_data/DB/db_discription.txt','w')
# tbl_num=11
# tbl_sizes=[100000,500000,1000000,2500000,5000000,7500000,10000000,12500000,15000000,17500000,20000000]
# NDVs=[100000,100000]
# max_repeatation=30
# st='table sizes: '
# st += ", ".join(str(x) for x in tbl_sizes)
# f.write(st+'\n')
# st='maximum NDVs for each join att: '
# st += ", ".join(str(x) for x in NDVs)
# f.write(st+'\n')
# DVs={}
# for inx in range(tbl_num):
#     ttt=time.time()
#     tbl=[]
#     att1 = np.random.choice(range(NDVs[0]), tbl_sizes[inx]).reshape(tbl_sizes[inx],1)
#     att2 = np.random.choice(range(NDVs[1]), tbl_sizes[inx]).reshape(tbl_sizes[inx],1)
#     tbl=np.append(att1,att2,axis=1)
#     tbl= np.unique(tbl, axis=0)
#     while len(tbl)<tbl_sizes[inx]:
#         l=tbl_sizes[inx]-len(tbl)
#         att1 = np.random.choice(range(NDVs[0]), l).reshape(l,1)
#         att2 = np.random.choice(range(NDVs[1]), l).reshape(l,1)
#         tbl_tmp = np.append(att1, att2, axis=1)
#         tbl_tmp = np.unique(tbl_tmp, axis=0)
#         tbl=np.append(tbl,tbl_tmp,axis=0)
#
#     MainTable=[]
#     for row in tbl:
#         rnd1=np.random.randint(1,max_repeatation)
#         for i in range(rnd1):
#             MainTable.append(row)
#     MainTable=np.array(MainTable).reshape(len(MainTable),2)
#     np.random.shuffle(MainTable)
#     print('writing in storage')
#     f.write('Table'+str(inx)+', att1 NDVs: '+ str(len(set(tbl[:, 0])))+', att2 NDVs : '+str(len(set(tbl[:, 1])))+', att1_att2 NDVs: '+str(tbl_sizes[inx])+', Max repeatation: '+str(max_repeatation)+', size of table: '+str(len(MainTable))+', time: '+str(time.time()-ttt))
#     f.write('\n')
#     np.savetxt('synthetic_data/DB/tbl_' + str(inx) + '.csv', MainTable.astype('str'), delimiter=",", fmt='%s %s')
#     print('Table {}, att1 NDVs: {}, att2 NDVs : {}, att1_att2 NDVs: {}, time: {}'.format(inx, len(set(tbl[:, 0])),
#                                                                                    len(set(tbl[:, 1])),tbl_sizes[inx],
#                                                                                    time.time() - ttt))
# f.close()
f= open('synthetic_data/DB3/db_discription.txt','w')
tbl_num=3
NDVs=[[2000,2000],[2000,2000],[2000,2000]]
tbl_sizes=10000
max_repeatation=5
st='table sizes: 1m'
f.write(st+'\n')
st='NDVs for each table: '
st += ", ".join(str(x) for x in NDVs)
f.write(st+'\n')
for inx in range(tbl_num):
    ttt=time.time()
    tbl=[]
    ndv=NDVs[inx]
    att1 =np.array( np.random.choice(range(ndv[0]), tbl_sizes).reshape(tbl_sizes,1)).astype(str)
    att2 =np.array( np.random.choice(range(ndv[1]), tbl_sizes).reshape(tbl_sizes,1)).astype(str)
    tbl=np.append(att1,att2,axis=1)
    tbl= np.unique(tbl, axis=0)
    while len(set(tbl[:,0]))<ndv[0] or len(set(tbl[:,1]))<ndv[1]:
        a1=list(set(range(ndv[0])) - set(tbl[:, 0]))
        a2=list(set(range(ndv[1])) - set(tbl[:, 1]))
        s=np.abs(len(a1)+len(a2))
        att1 = np.array(np.random.choice(a1, s).reshape(s, 1)).astype(str)
        att2 = np.array(np.random.choice(a2, s).reshape(s, 1)).astype(str)
        tbl_tmp = np.append(att1, att2, axis=1)
        tbl_tmp = np.unique(tbl_tmp, axis=0)
        tbl=np.append(tbl,tbl_tmp,axis=0)

    MainTable=[]
    for row in tbl:
        rnd1=np.random.randint(1,max_repeatation)
        for i in range(rnd1):
            MainTable.append(row)
    MainTable=np.array(MainTable).reshape(len(MainTable),2)
    np.random.shuffle(MainTable)
    print('writing in storage')
    # f.write('Table'+str(inx)+', att1 NDVs: '+ str(len(set(tbl[:, 0])))+', att2 NDVs : '+str(len(set(tbl[:, 1])))+', att1_att2 NDVs: '+str(tbl_sizes[inx])+', Max repeatation: '+str(max_repeatation)+', size of table: '+str(len(MainTable))+', time: '+str(time.time()-ttt))
    f.write('Table' + str(inx) + ', att1 NDVs: ' + str(len(set(tbl[:, 0]))) + ', att2 NDVs : ' + str(
        len(set(tbl[:, 1]))) + ', att1_att2 NDVs: ' + str(tbl_sizes) + ', Max repeatation: ' + str(
        max_repeatation) + ', size of table: ' + str(len(MainTable)) + ', time: ' + str(time.time() - ttt))
    f.write('\n')
    np.savetxt('synthetic_data/DB3/tbl_' + str(inx) + '.csv', MainTable.astype('str'), delimiter=",", fmt='%s %s')
    print('Table {}, att1 NDVs: {}, att2 NDVs : {}, att1_att2 NDVs: {}, time: {}'.format(inx, len(set(tbl[:, 0])),
                                                                                   len(set(tbl[:, 1])),tbl_sizes,
                                                                                   time.time() - ttt))
f.close()
