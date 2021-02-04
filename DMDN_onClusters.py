# Ali Mohammadi Shanghooshabad (Shanghoosh)
# University of Warwick
import math
import random
import time
import pickle
import tensorflow as tf
# tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import pandas as pd
import numpy as np
import OrdinalEncoding as oe
class DMDN:
    def __init__(self, batch_size=500,
                 learningRate=0.005,
                 no_layers=2,
                 no_hidden=16,
                 activationFunctionG=tf.nn.tanh):

        self.batch_size = batch_size
        self.learningRate = learningRate
        self.no_layers=no_layers
        self.no_hidden = no_hidden
        self.activationFunctionG = activationFunctionG
        tf.reset_default_graph()

    def generator(self, X, reuse=tf.AUTO_REUSE):
        if self.no_layers==5:
            with tf.variable_scope("Generator", reuse=reuse):
                h1 = tf.layers.dense(X, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h2 = tf.layers.dense(h1, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h3 = tf.layers.dense(h2, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h4 = tf.layers.dense(h3, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h5 = tf.layers.dense(h4, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                logits = tf.layers.dense(h5, self.no_out_bins)
                softMax = tf.nn.softmax(logits)
                return softMax, tf.random.categorical(tf.log(softMax), 1)
        elif self.no_layers == 4:
            with tf.variable_scope("Generator", reuse=reuse):
                h1 = tf.layers.dense(X, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h2 = tf.layers.dense(h1, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h3 = tf.layers.dense(h2, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h4 = tf.layers.dense(h3, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                logits = tf.layers.dense(h4, self.no_out_bins)
                softMax = tf.nn.softmax(logits)
                return softMax, tf.random.categorical(tf.log(softMax), 1)
        elif self.no_layers == 3:
            with tf.variable_scope("Generator", reuse=reuse):
                h1 = tf.layers.dense(X, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h2 = tf.layers.dense(h1, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h3 = tf.layers.dense(h2, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                logits = tf.layers.dense(h3, self.no_out_bins)
                softMax = tf.nn.softmax(logits)
                return softMax, tf.random.categorical(tf.log(softMax), 1)
        elif self.no_layers == 2:
            with tf.variable_scope("Generator", reuse=reuse):
                h1 = tf.layers.dense(X, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                h2 = tf.layers.dense(h1, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_uniform')
                logits = tf.layers.dense(h2, self.no_out_bins)
                softMax = tf.nn.softmax(logits)
                return softMax, tf.random.categorical(tf.log(softMax), 1)

    def cal_loss(self, probs, real_Ysamples):
        self.p = tf.gather(probs, real_Ysamples, batch_dims=1)
        self.result2 = -tf.log((self.p))
        return tf.reduce_mean(self.result2), probs

    def store(self, name):
        save_path = self.saver.save(self.sess, name)
        pickle_out = open(name+"_Y_ordinal" + ".pickle", "wb")
        pickle.dump(self.ordinalYencoding, pickle_out)
        pickle_out.close()

    def restore(self, name):
        self.saver = tf.train.Saver()
        # config = tf.ConfigProto()
        # jit_level = tf.OptimizerOptions.ON_1
        # config.graph_options.optimizer_options.global_jit_level = jit_level
        # session_conf = tf.ConfigProto(
        #     intra_op_parallelism_threads=3,
        #     inter_op_parallelism_threads=3)
        # self.sess = tf.Session(config=session_conf)
        self.sess = tf.Session()
        self.saver.restore(self.sess, name)
        print("Model "+name+" restored.")

    def restore_ordinalY(self,name):
        pickle_in = open(name + "_Y_ordinal.pickle", "rb")
        self.ordinalYencoding = pickle.load(pickle_in)
        self.no_out_bins=len(self.ordinalYencoding.main_dict)

    def predict_one(self,x_data_embedings, freq):
            x_data_embedings=np.array(x_data_embedings).reshape(1, self.no_in_bins)
            self.no_in_bins=len(x_data_embedings[0,:])

            probs_out = self.sess.run([self.probs], feed_dict={self.X: x_data_embedings})
            probs={}
            for i,p in enumerate(probs_out[0][0]):
                aaa=self.ordinalYencoding.inverse_transform([i])
                if isinstance(aaa, int):
                    aaa=str(aaa)
                elif isinstance(aaa, np.ndarray):
                    aaa=str(aaa[0])
                probs.update({aaa:(p*freq)})
            return probs

    def buildModel_largeData(self,no_out_bins,x_embeding):
        self.loop_allowed=True
        self.no_out_bins = no_out_bins
        self.no_in_bins = len(x_embeding[list(x_embeding.keys())[0]])

        self.Y = tf.placeholder(tf.int32, [None, 1])
        self.YY = tf.one_hot(self.Y, depth=self.no_out_bins)
        self.YY = tf.squeeze(self.YY, 1)
        self.X = tf.placeholder(tf.float32, [None, self.no_in_bins])
        self.probs, self.samples = self.generator(self.X)
        self.gen_loss, self.lgits = self.cal_loss(self.probs, self.Y)
        self.gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
        self.gen_step = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.gen_loss,var_list=self.gen_vars)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        return

    def fitModel_LargeData(self, data_chunk,dep_id=[0],indep_id=[1],max_noImprovement=100,ch_id=0,x_embeding=None, c_id=0,epoch=0,iteration=10):
        if self.loop_allowed==True:
            self.x_data = np.array(data_chunk[:, indep_id])
            self.y_data = np.array(data_chunk[:, dep_id]).reshape(len(data_chunk[:, dep_id]), 1)
            self.no_x_data = len(self.x_data)
            if ch_id == 0 and epoch==0:
                self.ordinalYencoding = oe.ordinalEncoding()
                self.y_data_ordinal_encoding = self.ordinalYencoding.fit_transform_largeData(self.y_data)
            else:
                self.y_data_ordinal_encoding = self.ordinalYencoding.fit_update_transform(self.y_data)
            self.x_data_embedings = []
            for x in self.x_data:
                self.x_data_embedings.append(list(x_embeding[str(x[0])]))
            self.x_data_embedings = np.array(self.x_data_embedings)
            lossa=[]
            for iter in range(iteration):
                for j in range(1+int(self.no_x_data / self.batch_size)):
                    # print(' method is 2 (split dataset), num of epoch is: ' + str(i) + 'the nn of the batch is: ' + str(j))
                    if j==int(self.no_x_data / self.batch_size):
                        x_batch = np.array(self.x_data_embedings[j * self.batch_size:self.no_x_data]).reshape(self.no_x_data-(j * self.batch_size), self.no_in_bins)
                        y_batch = np.array(self.y_data_ordinal_encoding[j * self.batch_size:self.no_x_data]).reshape(
                            self.no_x_data - (j * self.batch_size), 1)
                    else:
                        x_batch = np.array(self.x_data_embedings[j * self.batch_size:(j + 1) * self.batch_size]).reshape(self.batch_size, self.no_in_bins)
                        y_batch = np.array(
                            self.y_data_ordinal_encoding[j * self.batch_size:(j + 1) * self.batch_size]).reshape(
                            self.batch_size, 1)

                    _, gloss = self.sess.run(
                        [self.gen_step, self.gen_loss], feed_dict={self.Y: y_batch, self.X: x_batch})
                    lossa.append(gloss)
                s=''
                for ls in range(int(self.no_x_data / self.batch_size)):
                    idx=len(lossa)-ls-1
                    s+=str(lossa[idx])+'<---'
                # print("epoch: %d\t Generator loss: %.10f" % (i, lossa[last_epoch]))
                print(' clu_model: '+str(c_id)+' epoch: '+str(epoch)+' chunk: '+str(ch_id)+ ' iteration '+ str(iter)+':  '+s)
                if epoch==0 and ch_id==0:
                    self.best_loss =gloss
                    self.counter=0

                if self.best_loss>gloss:
                    self.best_loss=gloss
                    self.counter=0
                self.counter += 1
                if self.counter>max_noImprovement:
                    self.loop_allowed=False
            return gloss

    def cal_accuracy(self, main_data, indep_ids, dep_ids, test_size, x_embeding_dic):
        main_data = pd.DataFrame(main_data, columns=['src', 'dis'])
        print('started to cal accuracy')
        names = list(main_data.columns.to_numpy()[indep_ids])
        # names=list(main_data.columns.to_numpy()[indep_ids])
        # sample_df=main_data.sample(n = test_size,replace=False)   #  it is weighted sampling those x values that have high probabilities should be in prority
        uniq_counts = main_data.groupby(names).size().reset_index(name="freq")
        uniq_counts_arr = np.array(uniq_counts.values)
        uniq_counts_arr[:, -1] = uniq_counts_arr[:, -1] / main_data.count()[0]
        aaa = (uniq_counts_arr[:, 0].size)
        if aaa < test_size:
            test_size = aaa
            print('testsize was larger than distinct x values, we set it by max dist x values: ' + str(aaa))
        draw = np.random.choice(range(aaa), test_size, p=uniq_counts_arr[:, -1].astype('float64'), replace=False)
        sample_df = pd.DataFrame(uniq_counts_arr[draw, :-1], columns=names)
        count_x = []
        accuracy_all = []
        Precision_all = []
        Recall_all = []
        F1_Score_all = []

        import time
        st = time.time()
        ii = 0

        for index, row in sample_df[names].iterrows():
            print('on x with index= ' + str(ii))
            ii += 1
            row = pd.DataFrame([row])
            row = row.iloc[0, :]
            bbb = main_data.columns.to_numpy()[indep_ids]
            query = ''
            for att, value in zip(bbb, row.values):
                query += att + '=="' + str(value) + '" and '
            query = query[0:-4]
            rows = main_data.query(query)
            vars = indep_ids + dep_ids
            bbb = list(main_data.columns.to_numpy()[vars])
            x_group = rows.groupby(bbb).size().reset_index(name='Count')
            count_i = sum(x_group['Count'])
            count_x.append(count_i)
            vars = list(main_data.columns.to_numpy()[dep_ids])
            real_y = x_group[vars].values
            real_y = self.ordinalYencoding.transform(real_y)
            realy_dict = {}
            for y, c in zip(real_y, x_group.values):
                realy_dict.update({y: c[-1] / count_i})

            xx = np.array(row)[0]
            x_batch = np.array(x_embeding_dic[xx]).reshape(1, self.no_in_bins)
            probs_out = self.sess.run([self.probs], feed_dict={self.X: x_batch})
            fakey_dict = {}
            for i, p in enumerate(probs_out[0][0]):
                fakey_dict.update({i: p})
            notexist_yi = []

            def notexist(inp):
                notexist_yi.append(inp)
                return np.abs(inp)

            # d3 = {key: math.pow(realy_dict[key] - fakey_dict.get(key, 0),2) for key in realy_dict.keys()}
            # d3 = {key: math.pow(realy_dict[key] - fakey_dict[key], 2) if key in realy_dict else math.pow(fakey_dict[key], 2) for key in fakey_dict.keys()}
            # d4 = {key: fakey_dict[key] * np.abs(realy_dict[key] - fakey_dict[key]) if key in realy_dict
            #                     else notexist(fakey_dict[key]*np.abs(fakey_dict[key]-0))  for key in fakey_dict.keys()}

            TP = {}
            for key in fakey_dict.keys():
                if key in realy_dict:
                    if fakey_dict[key] >= realy_dict[key]:
                        TP.update({key: realy_dict[key]})
                    else:
                        TP.update({key: fakey_dict[key]})
                else:
                    TP.update({key: 0})
            TN = {}
            for key in fakey_dict.keys():
                if key in realy_dict:
                    if fakey_dict[key] >= realy_dict[key]:
                        TN.update({key: 1 - fakey_dict[key]})
                    else:
                        TN.update({key: 1 - realy_dict[key]})
                else:
                    TN.update({key: (1 - fakey_dict[key])})

            FP = {}
            for key in fakey_dict.keys():
                if key in realy_dict:
                    if fakey_dict[key] >= realy_dict[key]:
                        FP.update({key: fakey_dict[key] - realy_dict[key]})
                    else:
                        FP.update({key: 0})
                else:
                    FP.update({key: (fakey_dict[key])})
            FN = {}
            for key in fakey_dict.keys():
                if key in realy_dict:
                    if fakey_dict[key] >= realy_dict[key]:
                        FN.update({key: 0})
                    else:
                        FN.update({key: realy_dict[key] - fakey_dict[key]})
                else:
                    FN.update({key: 0})
            # accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same
            accuracy = (sum(TP.values()) + sum(TN.values())) / (
                        sum(FN.values()) + sum(FP.values()) + sum(TP.values()) + sum(TN.values()))
            Precision = sum(TP.values()) / (sum(TP.values()) + sum(FP.values()))
            Recall = sum(TP.values()) / (sum(TP.values()) + sum(FN.values()))
            # F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.
            # Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy,
            # especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost.
            # If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall
            F1_Score = 2 * (Recall * Precision) / (Recall + Precision)

            accuracy_all.append(accuracy * count_i)
            Precision_all.append(Precision * count_i)
            Recall_all.append(Recall * count_i)
            F1_Score_all.append(F1_Score * count_i)

        acc = sum(accuracy_all) / sum(count_x)
        pre = sum(Precision_all) / sum(count_x)
        rec = sum(Recall_all) / sum(count_x)
        f_score = sum(F1_Score_all) / sum(count_x)
        return f_score