# Model_Join  

Model_Join is a framework to join models which provides us the densities of single attributes and conditional probabilities of join attributes.
The framework is used when we have the models, but the data has been removed or we do not have access to the data.
Model_Join generates a uniform sample of the join result as a universal solution to do all the learning, knowledge discovery, data visualization or AQP over join results of deleted datasets.

# Setup

Download the project   
CD to the Model_Join folder  

Install Pyhton 3.6  
sudo apt install -y libpq-dev gcc python3-dev  
python3 -m venv venv  
source venv/bin/activate  
pip3 install -r requirements.txt  

# Code

To understand how the Model Join framework works please refer to the paper Model Join
For any join query a file like Join3Tables_tpcDS.py is built. The file employs all the existing models and generate the uniform sample of the join query.
Join3Tables_tpcDS.py is for the query Q3 of the paper. You can specify the number of process to run in parallel in the file.

In the case, the models for some datasets do not exists, you can use our code to generate the models.
For any join, we need:
i) We need the density of single join attributes in the to-be-joined datasets. We use dictionaries to keep the frequencies per distinct values in JAs.
ii) Based on the join order, we need models that provides the probability of right join attribute conditioned on the left join attribute. Said models are built as follows:

1- First we learn embeddings per JA in the middle datasets. 
To do so in our example Q3, copy csv files of Store_sales, Web_sales and store_returns of TPC-DS data from http://www.tpc.org/tpcds/  into the folder Model_Join/tpc_ds/3_way/
Note, you need to keep only the JAs (cutomer_key and item_key) in the files.
The file SkipGram_NegativeSampling.py is used to learn embeddings. Note, you need to specify the left and right JAs, number of hidden nodes, number of negative sampling, etc.

2- Once the embeddings were learned for both JAs in a datasets, the file clustering_embeddings.py is used to cluster the right JA. You can specify the number of the clusters.

3- To calculate the dictionaries of frequencies per JA, we use the file Frequences.py. Note, that this file also calculates the ground truth to evaluate the models. If you want to calculate frequencies for the datasets you do not have any model on, you should deactivate the calculating the ground truth.

4- Per cluster, a sub-model is learned. We use weighted_clusters_modeling.py to learn the sub-models. These sub-models  will be used a single model together.

Main.py is an example of using all the mentioned files for query Q3. 

DataBaseGenerator.py is the file you can generate the synthetic data.

Join_6way_tpc.py is the code for Q4 on TPC-DS data.

JoinTable_MP_3wayDB2.py is the framework code for Q1 on the synthetic data.

JoinTable_MP_DB2_t0t1t2t3t4.py is the framework code for Q2 on the synthetic data.






 






