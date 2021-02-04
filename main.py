import weighted_clusters_modeling as wc
import time
##################################################################  Learning embeddings#####################
import SkipGram_NegativeSampling as embed
#per intermediate table, the embeddings should be learned for both first and second JA
table='tpc_ds/3_way/store_sales.csv'
target='embeddings/tpc_ds/3_way/store_sales/Embedding_'
embed.main(table,target,0,1,30,20, False,0) #learn embedding for first JA on current table
embed.main(table,target,1,0,30,20, False,1) #learn embedding for the second JA on current table

# # # # # ################################################################# Clustering embeddings#####################
import clustering_embeddings as ce
out='clusters/tpc_ds/3_way/50c/'
f=open(out+'time0.txt','a+')
table='tpc_ds/3_way/store_sales.csv'
Y_emb='embeddings/tpc_ds/3_way/store_sales/Embedding__att1_embed.txt' # embeding for the second JA
tim=ce.main(table,Y_emb,out,1,1,50, False)
# tim = ce.main(table, Y_emb, out, i, 1, clu[cc], True)
f.write('time for overall clustering part of tbl'+str(tim)+'\n')
f.close()
# # # # # # # # # ######################################################### finding frequencies####################
import Frequences as fr
import time
out='clusters/tpc_ds/3_way/'
f=open(out+'time2.txt','a+')
t=time.time()
table='tpc_ds/3_way/web_sales.csv'
fr.main(table,out,0,False,False)
f.write('time for counting the att0 and att1 freqs and conditional freq on web sales: '+str(time.time()-t)+'\n')

table='tpc_ds/3_way/store_sales.csv'
fr.main(table,out,1,False, True)
f.write('time for counting the att0 and att1 freqs and conditional freq on store_sales: '+str(time.time()-t)+'\n')

table='tpc_ds/3_way/store_returns.csv'
fr.main(table,out,2,False,False)
f.write('time for counting the att0 and att1 freqs and conditional freq on  store_returns: '+str(time.time()-t)+'\n')

f.close()
#
# #
# # ######################################################################### learning models###################
# per cluster of each middle table we need a model
f = open('clusters/tpc_ds/3_way/50c/000_statistics.txt', "a+")
embed_add="embeddings/tpc_ds/3_way/store_sales/Embedding__att0_embed.txt"
cluster_add="clusters/tpc_ds/3_way/50c/tbl1_att1_reverseClusters_50.txt"
tbl="tpc_ds/3_way/store_sales.csv"
clusDictAdd='clusters/tpc_ds/3_way/50c/tbl1_att1clust_dict50.txt'
freqDictAdd='clusters/tpc_ds/3_way/50c/tbl1_att1freq_dict50.txt'
conditionalDict='clusters/tpc_ds/3_way/conditional_freq_tbl1.txt'
out='clusters/tpc_ds/3_way/50c/models/model_tbl1_'
num_layer=5
num_hidden=300
learning_rate=0.001
iteration=2
chunk_size=200000000
f.write('layers:'+str(num_layer)+', hidden:'+str(num_hidden)+', iter:'+str(iteration)+'\n')
all_t=time.time()
t1, t2, f1, test_size, testCount, intervals1, intervals2 = wc.main(embed_add, cluster_add, tbl, clusDictAdd, freqDictAdd, conditionalDict, out, num_layer,
                 num_hidden, iteration, learning_rate, chunk_size, False)
# t1, t2, f1, test_size, testCount, intervals1, intervals2 = wc.main(embed_add, cluster_add, tbl, clusDictAdd, freqDictAdd, conditionalDict, out, num_layer,
#                  num_hidden, iteration, learning_rate, chunk_size, True)
t3 = time.time() - all_t
f.write('#####just for testing ##### on table{} f score is:{}, test size is:{}, all predictions is:{}, time for training is:{}, time for testing is:{}, time for all is:{}, intervals on all is:{}, intervals on test ize is{}'.format(1,f1,test_size,testCount,t1,t2,t3,intervals1,intervals2))
# f.write(str(t1) + ',  ' + str(t2))
f.write('\n')
f.close()
