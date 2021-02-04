import pandas as pd
import numpy as np
import tensorflow as tf
import MDN as learner

embedings = pd.read_csv("embeddings/tpc-ds_small/tbl_5_att0_embed.txt",delimiter=',',header=None).values
embedings_dic1={}
for i in range(len(embedings[:,0])):
    embedings_dic1.update({str(int(embedings[i,0])):tuple(embedings[i,1:])})

embedings = pd.read_csv("embeddings/tpc-ds_small/tbl_5_att1_embed.txt",delimiter=',',header=None).values
embedings_dic2={}
for i in range(len(embedings[:,0])):
    embedings_dic2.update({str(int(embedings[i,0])):tuple(embedings[i,1:])})


data_df = pd.read_csv("synthetic_data/tpc-ds_small/tbl_5.csv",delimiter=',', usecols=[0,1,2])
data_df.columns = ['ja1','ja2','target']
data_df.dropna(inplace=True)
data=data_df.values
x_data = np.array(data[:, 0:2])
y_data = np.array(data[:, 2])
batchsize=5000
learningRate=0.0005
epoch=15000
no_noImprovement=50
no_hidden=300
no_layers=5
dropout=0.8
x_data_main=[]
for row in x_data:
    a=embedings_dic1[str(row[0])] + embedings_dic2[str(row[1])]
    x_data_main.append(a)
mdn=learner.MDN(x_data_main,y_data,40,'synthetic_data/tpc-ds_small/nonJA/',False,batchsize,learningRate,epoch,no_hidden,dropout,15)
mdn.fitModel()
