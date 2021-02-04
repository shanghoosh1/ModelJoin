import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sciclu
import time
def main(tableName,y_embeddAdd,outAdd,tbl_index,att_index,num_cluster1,compositeIn):
    t_all=time.time()
    # main_table="synthetic_data/DB1_2GB/tbl_1.csv"
    # table="embeddings/DB1_2GB/emb_tbl_1_att1.txt"
    # table_clusters_reverse="embeddings/DB1_2GB/emb_tbl_1_att1_reverseClusters100c.txt"
    # Clu_file='embeddings/DB1_2GB/clus_dic_tbl1_att0_100c.txt'
    # Freq_file='embeddings/DB1_2GB/freq_dic_tbl1_att0_100c.txt'
    main_table=tableName
    table=y_embeddAdd
    table_clusters_reverse=outAdd+'tbl'+str(tbl_index)+'_att'+str(att_index)+'_reverseClusters_'+str(num_cluster1)+'.txt'
    Clu_file=outAdd+'tbl'+str(tbl_index)+'_att'+ str(att_index)+'clust_dict'+str(num_cluster1)+'.txt'
    Freq_file=outAdd+'tbl'+str(tbl_index)+'_att'+str(att_index)+'freq_dict'+str(num_cluster1)+'.txt'

    chunksize=200000000
    indep_var_id=[0]
    dep_var_id=[1]
    num_cluster=num_cluster1

    embedings = pd.read_csv(table,delimiter=',',header=None).values
    k_means = sciclu.KMeans(n_clusters=num_cluster, max_iter=1000, n_jobs=5,verbose=1).fit(embedings[:,1:])
    labels=np.array(k_means.labels_).reshape(len(k_means.labels_),1)


    # f=open(table_clusters,'w+')
    # for i in range(len(embedings[:,0])):
    #     s=str(embedings[i,0])+', '+str(labels[i][0])
    #     f.write(s + '\n')
    # f.close()

    #reverse clustering
    embedings=np.append(labels,embedings,axis=1)
    clusters_data=[]
    cluster_sizes=[]
    for clu in range(num_cluster):
        temp = list((embedings[embedings[:,0] == clu])[:,1])
        clusters_data.append(temp)
        cluster_sizes.append(len(temp))
    new_clusters= [ [] for i in range(num_cluster) ]

    for i in range(num_cluster):
        for j in range(cluster_sizes[i]):
            c_id = random.randint(0, num_cluster-1)
            row=clusters_data[i].pop(0)
            new_clusters[c_id].append(row)
    new_clusters=np.array(new_clusters)
    f=open(table_clusters_reverse,'w+')
    for id,cluster in enumerate(new_clusters):
        c=np.array([id]*(len(cluster))).reshape(len(cluster),1)
        cluster1=np.append(np.array(cluster).reshape(len(cluster),1),c,axis=1)
        # cluster=np.append(cluster1,cluster[:,1],axis=1)
        for row in cluster1:
            s=str(row[0])
            for col in range(len(row)-1):
                s+=','+str(row[col+1])
            f.write(s + '\n')
    f.close()


    embedings = pd.read_csv(table_clusters_reverse,delimiter=',',header=None).values
    y_clusters={}
    for i in range(len(embedings[:,0])):
        if isinstance(embedings[i,0],float):
            y_clusters.update({str(int(embedings[i,0])):int(embedings[i,1])})
        else:
            y_clusters.update({str(embedings[i, 0]): int(embedings[i, 1])})
    # num_clusters=len(set(y_clusters.values()))
    ch_id=0

    x_num=1 # size of table
    x2indx = {}
    indx2x = {}
    index1=0
    freq_table=np.zeros((x_num,num_cluster))


    if not compositeIn:
        chunk = pd.read_csv(main_table, delimiter=',', usecols=[0, 1], engine='c')
        chunk.dropna(inplace=True)
        # chunk = chunk.sample(frac=1)
        chunk = chunk.astype(str)
    else:
        incol = ['a' + str(r) for r in range(4)]
        chunk = pd.read_csv(main_table, delimiter=',', usecols=[0,1,2,3], engine='c')
        chunk.columns = incol
        chunk.dropna(inplace=True)
        # chunk = chunk.sample(frac=1)
        chunk = chunk.astype(str)
        chunk['a0'] = chunk['a0'] + ',' + chunk['a1']+ ',' + chunk['a2']
        del chunk['a1']
        del chunk['a2']


    chunk.columns = ['src', 'dis']
    data = chunk.to_numpy()
    x_values=set(data[:,indep_var_id[0]])

    index_old=index1
    for word in x_values:
        if not str(word) in x2indx:
            x2indx[str(word)] = index1
            indx2x[index1] = str(word)
            index1+=1
    freq_table = np.pad(freq_table, ((0, index1-index_old), (0, 0)), 'constant', constant_values=(0))
    for row in data:
        clu=y_clusters[str(row[dep_var_id[0]])]
        freq_table[x2indx[row[indep_var_id[0]]], clu] += 1


    freq_table=np.delete(freq_table,len(freq_table)-1,axis=0)
    freq_dict={}
    clus_dict={}
    for i,row in enumerate(freq_table):
        clusters=[]
        percentages=[]
        sum1 = 0
        for j in range(len(row)):
            if row[j]>0:
                clusters.append(j)
                percentages.append(row[j])
                sum1+=row[j]
        # percentages/=sum1
        clus_dict.update({indx2x[i]:list(clusters)})
        freq_dict.update({indx2x[i]:list(percentages)})
    import json
    with open(Clu_file, 'w') as file:
        file.write(json.dumps(clus_dict))
    with open(Freq_file, 'w') as file:
        file.write(json.dumps(freq_dict))
    print('overall time')
    return (time.time()-t_all)
    #####################################################  we need to store the NDVs of y per cluster
    #
    # y_counts_cluster=np.array(np.unique(labels, return_counts=True))
    # y_counts_cluster=y_counts_cluster.astype('str')
    # y_counts_cluster_dic=dict(zip(y_counts_cluster[0,:],y_counts_cluster[1,:]))
    # with open('embeddings/NDVs_items_per_cluster.txt', 'w') as file:
    #     file.write(json.dumps(y_counts_cluster_dic))

