import random

import numpy as np
import tensorflow as tf
import pandas as pd
import pandas as pd

def main(tableName,targetAdd,left_id,right_id,num_hidden,num_negative,compositeIn, nameId):
    import time
    t_all=time.time()
    table = tableName
    output1 = targetAdd + '_att'+str(nameId)+'_embed.txt'
    output2=targetAdd+'_att'+str(nameId)+'_vis'
    output3=targetAdd+'_att'+str(nameId)+'_time_loss.txt'
    ff=open(output3,'w')
    left_varID=left_id
    if isinstance(right_id,int):
        rightVarID=[right_id]
    else:
        rightVarID=right_id
    All_in_Mem=True
    chunksize = 10000000000
    n_embedding = num_hidden
    n_sampled = num_negative # Number of negative labels to sample
    epochs = 100
    batch_size = 1000
    iteration = 1
    lossa = 0
    best_loss=100000.0
    NoChangeIteration=0
    NoChangeIteration_Stop=10000

    leftword2int = {}
    int2leftword = {}
    rightword2int = {}
    int2rightword = {}
    lw=set()
    rw=set()

    ch_id=0
    index1=0
    index2=0
    import time
    t=time.time()
    if not All_in_Mem:
        for chunk in pd.read_csv(table,
                                 chunksize=chunksize,
                                 iterator=True,delimiter=',', usecols=[0,1],engine='c'):
            print('preprocessing on chunk number: '+str(ch_id))
            ch_id+=1
            chunk.columns = ['src','dis']
            chunk.dropna(inplace=True)
            chunk=chunk.sample(frac=1)
            chunk=chunk.astype(str)
            data=chunk.to_numpy()
            lw=set(data[:,left_varID])
            for word in lw:
                if not word in leftword2int:
                    leftword2int[word] = index1
                    int2leftword[index1] = word
                    index1+=1

            for j in rightVarID:
                rw=set(data[:,j])
                for word in rw:
                    word+='_'+str(j)
                    if not word in rightword2int:
                        rightword2int[word] = index2
                        int2rightword[index2] = word
                        index2+=1
    else:
        if  not compositeIn:
            chunk=pd.read_csv(table, delimiter=',', usecols=[0, 1], engine='c')
            chunk.dropna(inplace=True)
            chunk = chunk.sample(frac=1)
            chunk = chunk.astype(str)
        else:
            if isinstance(left_varID, int):
                left_varID=[left_varID]
            allcol=left_varID+rightVarID
            incol=['a'+ str(r) for r in range(len(allcol))]
            chunk = pd.read_csv(table, delimiter=',', usecols=allcol, engine='c')
            chunk.columns =incol
            chunk.dropna(inplace=True)
            chunk = chunk.sample(frac=1)
            chunk = chunk.astype(str)
            if len(left_varID)==1 and len(rightVarID)==1:
                if left_varID[0]==0:
                    left_varID=0
                    rightVarID[0]=1
                else:
                    left_varID = 1
                    rightVarID[0] = 0
            if len(left_varID)>1:
                chunk['a0'] = chunk['a0']+',' +chunk['a1']+',' +chunk['a2']
                del chunk['a1']
                del chunk['a2']
                left_varID=0
                rightVarID[0]=1
            if len(rightVarID)>1:
                chunk['a0'] = chunk['a0'] + ',' + chunk['a1']+',' +chunk['a2']
                del chunk['a1']
                del chunk['a2']
                rightVarID[0] = 0
                rightVarID = [rightVarID[0]]
                left_varID=1


        chunk.columns = ['src', 'dis']
        data = chunk.to_numpy()
        lw = set(data[:, left_varID])
        for word in lw:
            if not word in leftword2int:
                leftword2int[word] = index1
                int2leftword[index1] = word
                index1 += 1

        for j in rightVarID:
            rw = set(data[:, j])
            for word in rw:
                word += '_' + str(j)
                if not word in rightword2int:
                    rightword2int[word] = index2
                    int2rightword[index2] = word
                    index2 += 1
    print(time.time()-t)
    left_vocab_size = len(leftword2int.keys()) # gives the total number of unique words
    right_vocab_size = len(rightword2int.keys()) # gives the total number of unique words



    import sys
    print(sys.getsizeof(chunk))
    print(sys.getsizeof(data))
    print(sys.getsizeof(rightword2int))
    print(sys.getsizeof(int2rightword))
    print(sys.getsizeof(leftword2int))
    print(sys.getsizeof(int2leftword))




    # function to convert numbers to one hot vectors
    def to_one_hot(data_point_index, vocab_size):
        temp = np.zeros(vocab_size)
        temp[data_point_index] = 1
        return temp

    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.int32, [None,1], name='inputs')
    #     labels = tf.placeholder(tf.int32, [None, None], name='labels')
        labels = tf.placeholder(tf.int32, [None,1], name='labels')


    with train_graph.as_default():
        embedding = tf.Variable(tf.random_uniform((left_vocab_size, n_embedding), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs) # use tf.nn.embedding_lookup to get the hidden layer output
        embed=tf.squeeze(embed,1)

    with train_graph.as_default():
        softmax_w = tf.Variable(tf.truncated_normal((right_vocab_size, n_embedding)))  # create softmax weight matrix here
        softmax_b = tf.Variable(tf.zeros(right_vocab_size), name="softmax_bias")  # create softmax biases here

        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(
            weights=softmax_w,
            biases=softmax_b,
            labels=labels,
            inputs=embed,
            num_sampled=n_sampled,
            num_classes=right_vocab_size,
            num_true=1)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    with train_graph.as_default():
        ## From Thushan Ganegedara's implementation
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
        valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
        valid_examples = np.append(valid_examples,
                                   random.sample(range(1000, 1000 + valid_window), valid_size // 2))

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embedding = embedding / norm
        # valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
        # similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


    keeplooping=True
    with tf.Session(graph=train_graph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        if not All_in_Mem:
            for e in range(epochs):
                if not keeplooping:
                    break
                ch_id = 0
                for chunk in pd.read_csv(table,
                                         chunksize=chunksize,
                                         iterator=True, delimiter=',', usecols=[0,1],engine='c'):
                    if not keeplooping:
                        break
                    ch_id += 1
                    chunk.columns = ['src', 'dis']
                    chunk.dropna(inplace=True)
                    chunk = chunk.sample(frac=1)
                    chunk = chunk.astype(str)
                    data = chunk.to_numpy()
                    x_train = []  # input word
                    y_train = []  # output word
                    for data_word in data:
                        left=data_word[left_varID]
                        right=data_word[rightVarID]
                        for j,wr in enumerate(right):
                            x_train.append(leftword2int[left])
                            wr+='_'+str(rightVarID[j])
                            y_train.append(rightword2int[wr])
                    # convert them to numpy arrays
                    x_train = np.asarray(x_train).reshape(len(x_train),1)
                    y_train = np.asarray(y_train).reshape(len(x_train),1)
                    for j in range(int(len(x_train) / batch_size)):
                        if j == int(len(x_train) / batch_size) - 1:
                            x_batch = np.array(x_train[j * batch_size:len(x_train)]).reshape(len(x_train)-(j * batch_size), 1)
                            y_batch = np.array(y_train[j * batch_size:len(x_train)]).reshape(len(x_train)-(j * batch_size), 1)
                        else:
                            x_batch = np.array(x_train[j * batch_size:(j + 1) * batch_size]).reshape(batch_size, 1)
                            y_batch = np.array(y_train[j * batch_size:(j + 1) * batch_size]).reshape(batch_size, 1)
                        start = time.time()
                        feed = {inputs: x_batch, labels: y_batch}
                        train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                        lossa += train_loss
                        # if iteration % 100 == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(e, epochs),
                              ", Chunk: {}".format(ch_id-1),
                              ", Batch_number: {}".format(j),
                              ", All iterations: {}".format(iteration),
                              ", Avg. Training loss: {:.8f}".format(train_loss / 100),
                              " ::::: {:.4f} sec/batch".format((end - start) / 100))
                        if train_loss<best_loss:
                            best_loss=train_loss
                            NoChangeIteration=0
                        else:
                            NoChangeIteration+=1
                            if NoChangeIteration>NoChangeIteration_Stop:
                                keeplooping=False
                                break
                        iteration += 1
                save_path = saver.save(sess, "checkpoints/text8.ckpt")
                embed_mat = sess.run(normalized_embedding)
        else:
            data = chunk.to_numpy()
            x_train = []  # input word
            y_train = []  # output word
            for data_word in data:
                left = data_word[left_varID]
                right = data_word[rightVarID]
                for j, wr in enumerate(right):
                    x_train.append(leftword2int[str(left)])
                    wr =str(wr)+ '_' + str(rightVarID[j])
                    y_train.append(rightword2int[wr])
            # convert them to numpy arrays
            x_train = np.asarray(x_train).reshape(len(x_train), 1)
            y_train = np.asarray(y_train).reshape(len(x_train), 1)
            for e in range(epochs):
                if not keeplooping:
                    break
                for j in range(int(len(x_train) / batch_size)):
                    if j == int(len(x_train) / batch_size) - 1:
                        x_batch = np.array(x_train[j * batch_size:len(x_train)]).reshape(len(x_train)-(j * batch_size), 1)
                        y_batch = np.array(y_train[j * batch_size:len(x_train)]).reshape(len(x_train)-(j * batch_size), 1)
                    else:
                        x_batch = np.array(x_train[j * batch_size:(j + 1) * batch_size]).reshape(batch_size, 1)
                        y_batch = np.array(y_train[j * batch_size:(j + 1) * batch_size]).reshape(batch_size, 1)
                    start = time.time()
                    feed = {inputs: x_batch, labels: y_batch}
                    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                    lossa += train_loss
                    # if iteration % 100 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          ", Chunk: {}".format(ch_id-1),
                          ", Batch_number: {}".format(j),
                          ", All iterations: {}".format(iteration),
                          ", Avg. Training loss: {:.8f}".format(train_loss / 100),
                          " ::::: {:.4f} sec/batch".format((end - start) / 100))
                    ff.write("Epoch {}/{}".format(e, epochs)+", Chunk: {}".format(ch_id-1)+ ", Batch_number: {}".format(j)+ ", All iterations: {}".format(iteration)+ ", Avg. Training loss: {:.8f}".format(train_loss / 100)+ " ::::: {:.4f} sec/batch".format((end - start) / 100))
                    ff.write('\n')
                    if train_loss<best_loss:
                        best_loss=train_loss
                        NoChangeIteration=0
                    else:
                        NoChangeIteration+=1
                        if NoChangeIteration>NoChangeIteration_Stop:
                            keeplooping=False
                            break
                    iteration += 1
                save_path = saver.save(sess, "checkpoints/text8.ckpt")
                embed_mat = sess.run(normalized_embedding)

    with train_graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        embed_mat = sess.run(embedding)

    f=open(output1,'w+')
    for i in range(left_vocab_size):
        s=''
        for j in range(n_embedding):
           s+=','+str(embed_mat[i,j])
        f.write((int2leftword[i]) + s + '\n')
    f.close()

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    viz_words = min(500,len(embed_mat))
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        # plt.annotate(int2leftword[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    plt.savefig(output2+'.pdf')
    # plt.show()
    print('#########################################  time for all steps')
    print(time.time()-t_all)
    ff.write(str(time.time()-t_all))
    ff.close()