import pandas as pd
import numpy as np
import tensorflow as tf
import DMDN_onClusters as learner
import time
import matplotlib.pyplot as plt
from collections import Counter
def main(embed_add,cluster_add,tbl,clusDictAdd,freqDictAdd,conditionalDict,out,num_lay,num_hid,itera,lr,ch_size, compositeIn):
    learningRate = lr
    epoch = 1
    iteration_on_chunk = itera
    no_noImprovement = 50
    no_hidden = num_hid
    no_layers = num_lay
    activationF = tf.nn.tanh
    batchsize = 1000
    chunksize = ch_size
    inputIds=[0]
    outputIds=[1]
    indep_var_id = [0]
    dep_var_id = [1]
    embed_use=True

    if not compositeIn:
        embedings = pd.read_csv(embed_add,delimiter=',',header=None).values
        embedings_dic={}
        for i in range(len(embedings[:,0])):
            if isinstance(embedings[i, 0], float):
                embedings_dic.update({str(int(embedings[i,0])):tuple(embedings[i,1:])})
            else:
                embedings_dic.update({str((embedings[i, 0])): tuple(embedings[i, 1:])})
    else:
        embedings = pd.read_csv(embed_add, delimiter=',', header=None).values
        embedings_dic = {}
        for i in range(len(embedings[:, 0])):
            kk=str(int(embedings[i, 0]))+','+str(int(embedings[i, 1]))+','+str(int(embedings[i, 2]))
            embedings_dic.update({kk: tuple(embedings[i, 3:])})

    embedings = pd.read_csv(cluster_add,delimiter=',',header=None).values
    y_clusters={}
    for i in range(len(embedings[:,0])):
        if isinstance(embedings[i, 0], float):
            y_clusters.update({str(int(embedings[i, 0])): int(embedings[i, 1])})
        else:
            y_clusters.update({str(embedings[i, 0]): int(embedings[i, 1])})
    num_clusters=len(set(y_clusters.values()))


    model_cluster=[[] for i in range(num_clusters)]


    All_inMemmory=True

    ttt=time.time()
    if All_inMemmory:
        if not compositeIn:
            data_chunk = pd.read_csv(tbl, delimiter=',', usecols=[0, 1], engine='c')
            data_chunk.dropna(inplace=True)
            # chunk = chunk.sample(frac=1)
            chunk = data_chunk.astype(str)
        else:
            incol = ['a' + str(r) for r in range(4)]
            data_chunk = pd.read_csv(tbl, delimiter=',', usecols=[0, 1, 2,3], engine='c')
            data_chunk.columns = incol
            data_chunk.dropna(inplace=True)
            # chunk = chunk.sample(frac=1)
            data_chunk = data_chunk.astype(str)
            data_chunk['a0'] = data_chunk['a0'] + ',' + data_chunk['a1']+ ',' + data_chunk['a2']
            del data_chunk['a1']
            del data_chunk['a2']

        data_chunk.columns = ['src', 'dis']
        data_chunk = data_chunk.to_numpy()
        data_chunk_c = np.empty([len(data_chunk),len(data_chunk[0])+1],dtype=object)
        for i,row in enumerate(data_chunk):
            data_chunk_c[i][0] = str(y_clusters[str(row[dep_var_id[0]])])
            for l in range(len(row)):
                data_chunk_c[i][l+1]=str(row[l])

        data_chunk=data_chunk_c
        data_chunk_c=None

    for c_id in range(num_clusters):
        no_out_bins = sum(value == c_id for value in y_clusters.values())
        no_in_bins = 0  # will be initialized later
        for epoch_id in range(epoch):
            ch_id=-1
            if not All_inMemmory:
                for data_chunk in pd.read_csv(tbl, chunksize=chunksize, iterator=True, delimiter=',',
                                         usecols=inputIds + outputIds, engine='c'):
                    ch_id+=1
                    data_chunk.columns = ['src', 'dis']
                    data_chunk.dropna(inplace=True)
                    data_chunk = data_chunk.sample(frac=1)
                    data_chunk = data_chunk.astype(str)
                    data_chunk = data_chunk.to_numpy()
                    data_chunk = data_chunk.astype('str')
                    c = []
                    for row in data_chunk:
                        clu = y_clusters[str(row[dep_var_id[0]])]
                        c.append(clu)
                    c = np.array(c).reshape(len(data_chunk[:, 0]), 1)
                    data_chunk = np.append(c, data_chunk, axis=1)

                    data = data_chunk[data_chunk[:, 0] == str(c_id)]
                    data = data[:, [1, 2]]
                    if ch_id==0 and epoch_id==0:
                        print('model for cluster'+ str(c_id))
                        model = learner.DMDN(batchsize, learningRate, no_layers, no_hidden, activationF)
                        model.buildModel_largeData(no_out_bins,embedings_dic)

                    last_loss = model.fitModel_LargeData(data,dep_var_id,indep_var_id,no_noImprovement,ch_id,embedings_dic,c_id,epoch_id,iteration_on_chunk)
                    print('End of chunk ######################  epoch {}, chunk {}, loss {}'.format(str(epoch_id),str(ch_id),str(last_loss)))
                    f_score = model.cal_accuracy(data, [0], [1], 100, embedings_dic)
                    print('f score is : ' + str(f_score))
                    model_cluster[c_id]=model
                    model.store(out + str(c_id))
            else:
                    ch_id=0
                    data = data_chunk[data_chunk[:, 0] == str(c_id)]
                    data = data[:, [1, 2]]
                    if epoch_id == 0:
                        print('model for cluster' + str(c_id))
                        model = learner.DMDN(batchsize, learningRate, no_layers, no_hidden, activationF)
                        model.buildModel_largeData(no_out_bins, embedings_dic)

                    last_loss = model.fitModel_LargeData(data, dep_var_id, indep_var_id, no_noImprovement, ch_id,
                                                         embedings_dic, c_id, epoch_id, iteration_on_chunk)

                    print('End of chunk ######################  epoch {}, loss {}'.format(str(epoch_id), str(last_loss)))
                    f_score = model.cal_accuracy(data, [0], [1], 100,embedings_dic)
                    print('f score is : '+str(f_score))
                    model.store(out + str(c_id))
                    # model_cluster[c_id] = model
    print('time for learning models of clusters: '+ str(time.time()-ttt))
    WholeModelingTime=time.time()-ttt

    import json
    with open(clusDictAdd, 'r') as file:
        clus_dict=(json.load(file))
    with open(freqDictAdd, 'r') as file:
        freq_dict=(json.load(file))
    with open(conditionalDict, 'r') as file:
        conditional_freq = (json.load(file))
    conditional_freq = np.array(conditional_freq)
    testX=conditional_freq[:,0]
    conditional_freq=conditional_freq[:,1]
    cluster_model1 = [[] for i in range(num_clusters)]
    t1 = time.time()
    for i in range(num_clusters):
        model = learner.DMDN(no_layers=5, no_hidden=no_hidden)
        model.restore_ordinalY(out + str(i))
        no_out_bins = sum(value == i for value in y_clusters.values())
        model.buildModel_largeData(no_out_bins, embedings_dic)
        model.restore(out + str(i))
        cluster_model1[i] = model
    LoadingTime=time.time()-t1
    # testX=np.array(conditional_freq.keys()).reshape(len(conditional_freq),1)
    FP = {}
    TN = {}
    TP = {}
    FN = {}
    t2=time.time()
    sum1=0
    iii=0
    ti=[]
    for ii,key in enumerate(testX):

        iii += 1
        print('testing on x ' + str(ii))
        clu = list(clus_dict[str(key)])
        freq = list(freq_dict[str(key)])
        fakey_dict={}
        for i, inx in enumerate(clu):
            m =cluster_model1[inx]
            if not compositeIn:
                 key = str(key).replace(',', '')
            probs = m.predict_one(list(embedings_dic[str(key)]), freq[i])
            fakey_dict = {**fakey_dict, **probs}
            sum1+=sum(freq)
            # sum2=sum(list(fakey_dict.values()))
            realy_dict=conditional_freq[ii]
            tt=time.time()
        for key in fakey_dict.keys():
            if key in realy_dict:
                if fakey_dict[key] >= realy_dict[key]:
                    if key in TP:
                        TP[key]=TP[key]+realy_dict[key]
                    else:
                        TP.update({key: realy_dict[key]})

                    if key in TN:
                        TN[key] = TN[key] + 1 - fakey_dict[key]
                    else:
                        TN.update({key: 1 - fakey_dict[key]})

                    if key in FP:
                        FP[key] = FP[key] + fakey_dict[key] - realy_dict[key]
                    else:
                        FP.update({key: fakey_dict[key] - realy_dict[key]})

                    if key in FN:
                        FN[key] = FN[key] + 0
                    else:
                        FN.update({key: 0})

                else:
                    if key in TP:
                        TP[key] = TP[key] + fakey_dict[key]
                    else:
                        TP.update({key: fakey_dict[key]})

                    if key in TN:
                        TN[key] = TN[key] +  1-realy_dict[key]
                    else:
                        TN.update({key: 1-realy_dict[key]})

                    if key in FP:
                        FP[key] = FP[key] + 0
                    else:
                        FP.update({key: 0})

                    if key in FN:
                        FN[key] = FN[key] + realy_dict[key] - fakey_dict[key]
                    else:
                        FN.update({key: realy_dict[key] - fakey_dict[key]})

            else:
                if key in TP:
                    TP[key] = TP[key] + 0
                else:
                    TP.update({key: 0})

                if key in TN:
                    TN[key] = TN[key] + 1 - fakey_dict[key]
                else:
                    TN.update({key: 1 - fakey_dict[key]})

                if key in FP:
                    FP[key] = FP[key] + fakey_dict[key]
                else:
                    FP.update({key: fakey_dict[key]})

                if key in FN:
                    FN[key] = FN[key] + 0
                else:
                    FN.update({key: 0})

        print('time for Tp of one x: '+ str(time.time()-tt))

    Precision = sum(TP.values()) /( sum(TP.values()) + sum(FP.values()))
    Recall = sum(TP.values()) / (sum(TP.values()) + sum(FN.values()))
    F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
    print(F1_Score)
    print('time for all test=' + str(time.time()-t2))
    WholeTestingTime=time.time()-t2
    import json
    with open(out+'_TP', 'w') as file:
        file.write(json.dumps(TP))
    with open(out+'_TN', 'w') as file:
        file.write(json.dumps(TN))
    with open(out+'_FP', 'w') as file:
        file.write(json.dumps(FP))
    with open(out+'_FN', 'w') as file:
        file.write(json.dumps(FN))
    intervals=1.96 * np.sqrt((F1_Score * (1 - F1_Score)) / sum1)
    intervals2 = 1.96 * np.sqrt((F1_Score * (1 - F1_Score)) / len(testX))
    return WholeModelingTime,WholeTestingTime,F1_Score,len(testX),sum1,intervals,intervals2
