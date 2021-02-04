

###########  be careful about deleting null values. variables effect each other in frequency calculating

import random
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
def main(table,outAdd, tbl_index,compositeIn, evaluation):

    out1 = outAdd + 'freq_tbl' + str(tbl_index) + '_att0.txt'
    out2 = outAdd + 'freq_tbl' + str(tbl_index) + '_att1.txt'
    out3=outAdd + 'conditional_freq_tbl' + str(tbl_index) + '.txt'
    ch_id=0
    left_varID=0
    rightVarID=[1]
    left_freq_dict={}
    rightFreq_dict={}
    chunksize=200000000
    testsize = 1000

    #
    if not compositeIn:
        chunk = pd.read_csv(table, delimiter=',', usecols=[0, 1], engine='c')
        chunk.dropna(inplace=True)
        # chunk = chunk.sample(frac=1)
        chunk = chunk.astype(str)
    else:
        incol = ['a' + str(r) for r in range(4)]
        chunk = pd.read_csv(table, delimiter=',', usecols=[0,1,2,3], engine='c')
        chunk.columns = incol
        chunk.dropna(inplace=True)
        # chunk = chunk.sample(frac=1)
        chunk = chunk.astype(str)
        chunk['a0'] = chunk['a0'] + ',' + chunk['a1']+ ',' + chunk['a2']
        del chunk['a1']
        del chunk['a2']

    chunk.columns = ['src', 'dis']
    data = chunk.to_numpy()

    left_freq_temp=np.array(np.unique(data[:,left_varID], return_counts=True))
    left_freq_dict_tmp=dict(zip(left_freq_temp[0,:],left_freq_temp[1,:]))
    left_freq_dict = Counter(left_freq_dict) + Counter(left_freq_dict_tmp)
    rightFreq=np.array(np.unique(data[:,rightVarID], return_counts=True))
    rightFreq_dict_tmp=dict(zip(rightFreq[0,:],rightFreq[1,:]))
    rightFreq_dict = Counter(rightFreq_dict) + Counter(rightFreq_dict_tmp)
    import json
    with open(out1, 'w') as file:
        file.write(json.dumps(left_freq_dict))
    with open(out2, 'w') as file:
        file.write(json.dumps(rightFreq_dict))

    ################################################  prepare 1000 uniformly chosen x values and their probs for evaluation step
    if evaluation:
        import json
        with open(out2, 'r') as file:
            rightFreq_dict=(json.load(file))
        with open(out1, 'r') as file:
            left_freq_dict=(json.load(file))

        l_keys=list(left_freq_dict.keys())
        l_values=list(left_freq_dict.values())
        aaa=sum(l_values)
        l_values= [r/aaa for r in l_values]
        aaa=sum(l_values)
        draw = np.random.choice(range(len(l_keys)), testsize,p=l_values)
        X_samples=np.array(l_keys)[draw]

        freq_samples=np.zeros([len(X_samples),len(list(rightFreq_dict.keys()))],dtype='int')
        freq_list=[[[],{}] for i in range(testsize)]
        for index, row1 in enumerate(X_samples):
            freq_list[index][0]=row1

        for index, row1 in enumerate(X_samples):
            print('on x with index= ' + str(index))
            row = pd.DataFrame([row1])
            row = row.iloc[0, :]
            query = ''
            for att, value in zip(['src'], row.values):
                query += att + '=="' + str(value) + '" and '
            query = query[0:-4]
            rows = chunk.query(query)
            bbb = list(chunk.columns.to_numpy())
            x_group = rows.groupby(bbb).size().reset_index(name='Count')
            count_i = sum(x_group['Count'])
            real_y = x_group.values
            dic={}
            for y in real_y:
                dic.update({y[1]: y[-1] })
            freq_list[index][1] = Counter(freq_list[index][1]) + Counter(dic)

        with open(out3, 'w') as file:
            file.write(json.dumps(freq_list))



























    ##################################################################################
    #
    # table="web_sales.dat"
    # ch_id=0
    # left_freq_dict={}
    # chunksize=5000000
    # for chunk in pd.read_csv(table,
    #                          chunksize=chunksize,
    #                          iterator=True,delimiter='|', usecols=[4],engine='c'):
    #     print('preprocessing on chunk number: '+str(ch_id))
    #     ch_id+=1
    #     chunk.dropna(inplace=True)
    #     chunk=chunk.sample(frac=1)
    #     chunk=chunk.astype(str)
    #     data=chunk.to_numpy()
    #     left_freq = np.array(np.unique(data[:, 0], return_counts=True))
    #     left_freq_dict_tmp = dict(zip(left_freq[0, :], left_freq[1, :]))
    #     left_freq_dict = Counter(left_freq_dict) + Counter(left_freq_dict_tmp)
    # import json
    # with open('embeddings/Freq_web_sales_customers.txt', 'w') as file:
    #     file.write(json.dumps(left_freq_dict))
    #
    # table="catalog_sales.dat"
    # ch_id=0
    # left_freq_dict={}
    # chunksize=5000000
    # for chunk in pd.read_csv(table,
    #                          chunksize=chunksize,
    #                          iterator=True,delimiter='|', usecols=[15],engine='c'):
    #     print('preprocessing on chunk number: '+str(ch_id))
    #     ch_id+=1
    #     chunk.dropna(inplace=True)
    #     chunk=chunk.sample(frac=1)
    #     chunk=chunk.astype(str)
    #     data=chunk.to_numpy()
    #     left_freq = np.array(np.unique(data[:, 0], return_counts=True))
    #     left_freq_dict_tmp = dict(zip(left_freq[0, :], left_freq[1, :]))
    #     left_freq_dict = Counter(left_freq_dict) + Counter(left_freq_dict_tmp)
    # import json
    # with open('embeddings/Freq_catalog_sales_items.txt', 'w') as file:
    #     file.write(json.dumps(left_freq_dict))

