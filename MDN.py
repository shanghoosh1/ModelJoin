import math
import tensorflow as tf
import pandas as pd
import category_encoders as ce
import numpy as np
import matplotlib.pyplot as plt
class MDN:
    def __init__(self,x_data,y_data,no_input=100, outaddress=None, output_normalization=True, batch_size = 500,learningRate=0.005,
                 epoch=20000,
                 no_hidden=16,
                 dropoutRate=0.5, mixtures=1):
            self.no_in_bins = no_input
            self.x_data=x_data
            self.y_data=y_data
            self.out=outaddress
            self.batch_size = batch_size
            self.learningRate = learningRate
            self.epoch = epoch
            self.no_hidden = no_hidden
            self.dropoutRate = dropoutRate
            self.no_mix=mixtures
            self.outputNorm=output_normalization
            self.oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
            self.no_out_bins = mixtures*3
            self.activationFunctionG=tf.nn.relu
    def preprocessing(self):

        self.no_x_data=len(self.x_data)
        self.minY = min(self.y_data)
        self.maxY = max(self.y_data)
        self.y_data = (self.y_data - self.minY) / (self.maxY - self.minY)
    def generator(self,X):
        with tf.variable_scope("Generator"):
                h1 = tf.layers.dense(X, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_normal')
                # hidden_layer1Dropout = tf.nn.dropout(h1, keep_prob=self.dropoutRate)
                h2 = tf.layers.dense(h1, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_normal')
                # hidden_layer2Dropout = tf.nn.dropout(h2, keep_prob=self.dropoutRate)
                h3 = tf.layers.dense(h2, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_normal')
                h4 = tf.layers.dense(h3, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_normal')
                h4 = tf.layers.dense(h4, self.no_hidden, activation=self.activationFunctionG,
                                     kernel_initializer='glorot_normal')
                logits = tf.layers.dense(h4, self.no_out_bins)

                return logits
    def get_parameters(self, inputLogits):
        pii = tf.placeholder(dtype=tf.float32, shape=[None, self.no_mix])
        sigmai = tf.placeholder(dtype=tf.float32, shape=[None, self.no_mix])
        mui = tf.placeholder(dtype=tf.float32, shape=[None, self.no_mix])
        pii, sigmai, mui = tf.split(inputLogits, [self.no_mix, self.no_mix, self.no_mix], 1)
        max_pi1 = tf.reduce_max(pii, 1, keep_dims=True)
        out_pi1 = tf.subtract(pii, max_pi1)
        out_pi1 = tf.exp(out_pi1)
        normalize_pi1 = tf.reciprocal(tf.reduce_sum(out_pi1, 1, keep_dims=True))
        out_pi1 = tf.multiply(normalize_pi1, out_pi1)
        sigmai = tf.exp(sigmai)
        return out_pi1,mui,sigmai

    def tf_normal(self,y, mu, sigma):
        result = tf.subtract(y, mu)
        result = tf.multiply(result, tf.reciprocal(sigma))
        result = -tf.square(result) / 2
        return tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * self.oneDivSqrtTwoPI

    def get_lossfunc(self,pi, mu, sigma, y):
        result = self.tf_normal(y, mu, sigma)
        result = tf.multiply(result, pi)
        self.result7878 = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(self.result7878)
        return tf.reduce_mean(result)

    def buildmodel(self):

        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.X = tf.placeholder(tf.float32, [None, self.no_in_bins])

        self.logits= self.generator(self.X)
        self.pi,self.mu,self.sigma=self.get_parameters(self.logits)
        self.gen_loss= self.get_lossfunc(self.pi,self.mu,self.sigma, self.Y)
        self.gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
        self.gen_step = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.gen_loss,var_list=self.gen_vars)

    def fitModel(self):
        self.preprocessing()
        self.buildmodel()
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        f = open(self.out+'loss_logs.csv', 'w+')
        lossa = []
        for i in range(self.epoch + 1):
            for j in range(1+int(self.no_x_data / self.batch_size)):
                # print(' method is 2 (split dataset), num of epoch is: ' + str(i) + 'the nn of the batch is: ' + str(j))
                if j==int(self.no_x_data / self.batch_size):
                    x_batch = np.array(self.x_data[j * self.batch_size:self.no_x_data]).reshape(self.no_x_data-(j * self.batch_size), self.no_in_bins)
                    y_batch = np.array(self.y_data[j * self.batch_size:self.no_x_data]).reshape(self.no_x_data - (j * self.batch_size), 1)
                else:
                    x_batch = np.array(self.x_data[j * self.batch_size:(j + 1) * self.batch_size]).reshape(self.batch_size, self.no_in_bins)
                    y_batch = np.array(
                        self.y_data[j * self.batch_size:(j + 1) * self.batch_size]).reshape(
                        self.batch_size, 1)
                _, gloss,res= self.sess.run([self.gen_step, self.gen_loss,self.result7878], feed_dict={self.Y: y_batch, self.X: x_batch})

                lossa.append(gloss)
                print("epoch: %d\t Generator loss: %.4f" % (i, gloss))
            if i%50==0:
                self.conditional_sample(x_batch,y_batch,i)
                if gloss==np.nan:
                    exit()
    def conditional_sample(self,x_test,y_test,itr):
        out_pi, out_mu, out_sigma, out_logits,X= self.sess.run([ self.pi,self.mu,self.sigma,self.logits,self.X], feed_dict={self.X: x_test})
        output = []
        for i in range(0, len(x_test)):
                rnd = np.random.rand()  # initially random [0, 1]
                idx = self.get_pi_idx(rnd, out_pi[i])
                mu = out_mu[i, idx]
                std = out_sigma[i, idx]
                rn = np.random.randn()  # normal random matrix (0.0, 1.0)
                y_new = mu + rn * std
                output.append([x_test[i], y_new])
        output=np.array(output)
        y_test1=y_test[:150]
        output1=output[:150]
        sumP=[]
        for i in range(len(out_mu[0])):
            sumP.append([i,sum(out_pi[:,i])])
        sumP = sorted(sumP, key = lambda x: (x[1]),reverse=True)

        for ii in range(len(out_mu[0])):
            plt.plot(range(len(y_test1)), out_pi[0:150, sumP[ii][0]], color='blue', label='Prob. of 1st gaussian')
            plt.plot(range(len(y_test1)), out_mu[0:150,sumP[ii][0]], label='1st gaussian',color='red' )

            # plt.plot(range(len(y_test1)), out_mu[0:200, sumP[1][0]], label='2nd gaussian',color='blue' )
            # plt.plot(range(len(y_test1)), out_pi[0:200, sumP[1][0]],color='gray', label='Prob. of 2nd gaussian')
            plt.plot(range(len(y_test1)), y_test1, 'g^', label='Actual Data points')
            plt.plot(range(len(y_test1)), output1[:, 1], 'b*',label='Generated points')
            # plt.plot(range(len(y_test1)), y_test1, 'g^')
            plt.xlabel('Indep. values')
            plt.ylabel('Normalized dep. values & Probs')
            # plt.legend(loc='upper right')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                       ncol=2, mode="expand", borderaxespad=0.)
            plt.savefig(self.out+'output'+str(itr)+'_'+str(ii) )
            # plt.show()
            plt.close()

        print('######################################################################'+str(self.calculate_err(self.x_data,self.y_data)))
        return output
    def get_pi_idx(self,x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        print('error with sampling ensemble')
        return -1
    def calculate_err(self,x_test,y_test):
        out_pi, out_mu, out_sigma, out_logits,X= self.sess.run([ self.pi,self.mu,self.sigma,self.logits,self.X], feed_dict={self.X: x_test})
        output = []
        for i in range(0, len(x_test)):
                rnd = np.random.rand()  # initially random [0, 1]
                idx = self.get_pi_idx(rnd, out_pi[i])
                mu = out_mu[i, idx]
                std = out_sigma[i, idx]
                rn = np.random.randn()  # normal random matrix (0.0, 1.0)
                y_new = mu + rn * std
                output.append(y_new)
        output=np.array(output)
        s=0
        for i in range(len(y_test)):
           s+=abs((output[i])-(y_test[i]))/max(abs(y_test[i]),abs(output[i]))

        # print(s/len(y_test))
        return s/len(y_test)
