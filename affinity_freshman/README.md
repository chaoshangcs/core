###Quick Introduction to Affinity  
_The aim of the resources on this page is to allow anyone, even without specific machine learning background, to quickly get up to speeed with Affinity Core virtual screening engine. The estimated time for completions is under two weeks._

![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/affinity_how2.png)
####Background Readings
_If you are familiar with all of the concepts the list: weights, biases, activation function, ReLU, softmax, convolution, pooling, layers of depth, batch, gradient descent, backpropagation (and chain rule), AdamOptimizer, please feel free to skip to the next section._  

CS231n "Convolutional Neural Networks for Visual Recognition"  
http://cs231n.github.io/  
Please, read through  
Module 1, Neural Networks      
Module 2, Convolutional Neural Networks   


####TensorFlow tutorials to complete
_If you are familiar with all of the concepts in this list: tensor graph, session, tf.global_variable_initializer, tf.train.coordinator, tf.train.start_queue_runners, tf.nn.sparse_softmax_cross_entropy_with_logits, tf.saver, tf.summary, tf.summary.FileWriter, tf.name_scope, please, feel free to skip to the next section._


[MNIST](https://www.tensorflow.org/tutorials/mnist/beginners/)  
[Deep MNIST](https://www.tensorflow.org/tutorials/mnist/pros/)  
[Understandig TensorFlow's workflow](https://www.tensorflow.org/tutorials/mnist/tf/)  
[CIFAR10](https://www.tensorflow.org/tutorials/deep_cnn/)  
[Deep Generative Adversarial Models](https://github.com/carpedm20/DCGAN-tensorflow)  

####Introduction to Affinity Core

_Structure based virtual screening is an approach that allows to retrieve a very small percent, usually few dozens of molecules, from the large database, of millioons of chemical structures. The process can be imagined as a google search for a flexible key (ligand) with a 3D image of a rigid lock (receptor,protein). Search can be broken into two parts. Since the most optimal relative position of the drug and protein is not known, it has to be estimated (docking). Afterwards, many static protein-ligand complexes have to be ranked by their predicted relative binding affinity (sorting). Usually, 25,000-200,000 pose evaluations are done during docking, and a single pose evaluation is done during ranking. Because Tesla K80 GPU can only evaluate 100-200 images/second, position search for a single ligand may take anywhere between 3 and 35 minutes, docking the average-size database of 1,000,000 of molecules may take 1.2 GPU years. In this example we only apply the network to the previously docked with AutoDock Smina positions, IE: ranking._

[![network_search](https://github.com/mitaffinity/core/blob/master/misc/search.png)](https://youtu.be/qHxrW6NUjkU)

####Step 1: teaching the network
You will need four scripts, and the database
```
av4_networks.py
av4_main.py
av4_input.py
av4_utils.py
labeled_av4
```
where `labeled_av4` is an already prepared database of the ligands and proteins in av4 binary format.


```
av4_networks.py 
# is a library of many different network architectures
# each of the networks accepts batch of images, and outputs batch of unscaled probabilities (logits)
#
# crucial part 1: the network itself
# convolutional layers IE: tf.nn.conv3d
# pooling layers IE: tf.nn.max_pool3d or tf.nn.avg_pool3d
# Rectifier Linear Regression Units, or ReLUs IE: tf.nn.relu
#
# crucial part 2: rules for variable initialization
# IE: bias_variable 
# tf.constant(0.01, shape=shape)
# or weight variable
# tf.truncated_normal(shape, stddev=0.005)
# initial weights and biases for trainable variables are usually initialized with small random positive 
# values. Deep networks can easily run out of control and owerflow the floats in higher layers 
# if not initialized properly.
# That's why it's important to initialize variables very accurately
# especially deep networks can be hierarchically constructed 
# when the new layer(s) is added to the top of existing trained network
#
# crucial part 3: variable summaries
# IE: tf.summary.histogram, tf.summary.scalar, 
# and tf.name_scope (groups variables together under a common name)
# variable summaries are written in a separate file and help to monitor the state and evolution 
# of the network during training or testing


av4_input.py
# is script that reads and indexes the database in av4 format, and creates batches of images
# in 3D that can be fed to the network
# av4 database consists of thousands of folders - one for each protein
# each protein can have many ligands
# each ligand av4 file can have many positions (frames), and every frame has it's label
# the name of each folder IE 1QGT, 4G93 each correspond to a particular PDB id  
# original PDB file with coordinates can be found at http://www.rcsb.org/
#
# crucial part 1: index_the_database_into_queue
# crawls of all the folders in the database and creates tensor of filenames
#
# crucial part 2: read_receptor_and_ligand
# reads a single example of receptor and ligand from the database
# returns coordinates and name (label) of every atom (only one frame depending on epoch counter)
#
# crucial part 3: convert_protein_and_ligand_to_image
# creates an image (sparse or dense) from input atom coordinates
# sparse image roughly can be described as array like
# [[atom_tag1,x1,y1,z1]
#  [atom_tag2,x2,y2,z2]
#  [atom_tag3,x3,y3,z3]]
# dense image is a 3D cube filled with zeros and float numbers
# both PDB and av4 store data in sparse representations
# network, however, needs a 3D variant to do convolutions
#
# crucial part 4: tf.train.batch (for three reasons)
# reason 1:
# it runs multiple independent threads of image creation
# reason 2:
# it batches images together
# single images are small and tensor operations in them (such as convolution or pooling)
# are not efficient. 
# reason 3:
# gradient descent optimizer gets much better gradient from multiple images. 
# Ideally, every single gradient descent step would be applied to a representative sample 
# of a whole database; this is better achievable with larger batches


# av4_main.py
# is a main script that assembles all of the parts together 
# 
# crucial part 1: FLAGS
# class that stores many global parameters of the script
# such as FLAGS.pixels_size - determines the size of the pixel generated by av4_input
# 
# crucial part 2: train
# assembles all of the parts together:
# 1. input image creation pipeline              IE: ....something = image_and_label_queue 
# 2. network that makes predictions             IE: intuit_net or nilai_net
# 3. cost function to minimize                  IE: tf.nn.sparse_softmax_cross_entropy_with_logits
# 4. optimizer(applies gradient descent steps)  IE: tf.train.AdamOptimizer
# 5. run while loop pipe the tensor graph       IE: sess.run(train_step_run)
# graph initialization commands:
# 6. initializer of variables from placeholders IE: tf.global_variable_initializer 
# 7. fill the graph skeleton with data inputs   IE: tf.train.start_queue_runners
# 8. also has saver of variable states          IE: tf.train.saver
# 9. variable summaries to visualize            IE: tf.summaries.writer
# 10. an epoch counter that counts every read through all proteins in database as one epoch

av4_utils.py
# stores various utilities to support functions that are not natively present in TensorFlow
``` 
here is how a typical session on our Amazon graphical instance with K80 GPU would look like:

```bash
# log into our remote machine 
# email maksym to get the key
$ ssh -i P2_key.pem ubuntu@awsinstance.com
# every member of the group should have his or her folder
$ cd maksym
# clone affinity core into your working directory 
ubuntu@ip-172-31-4-5:~/maksym$ git clone https://github.com/mitaffinity/core.git  
$ cd core 
$ python av4_main.py
# does not work; the database is empty
# point the script to the location of the database
$ vi (or any other command line text file editor; some people like nano) 
# the database has already been downloaded to the instance
# change the database path under flags to   
# /home/ubuntu/common/data/labeled_av4  
$ python av4_main.py  

# does not work; needs latest tensorflow  
# tensorflow12 is hidden in an envoronmental variable 
$ source $TF12  
# if you are interested what $TF12 it is:  
$ echo $TF12 
/home/ubuntu/common/venv/tf12/bin/activate  

# start training
$ python av4_main.py 
# seems to work, now it's time to launch this process for a while
# the key is to launch it on the background, so it does not die when you log off
# from your remote host. Use the '&' sign
$ python av4_main.py &  
# now background process will persist when you exit the session  

# now the nasty problem: TensorFlow tends not to die and hog on the GPU even after it's been terminated
# also, there are many of us using GPU instance at the same time, but with TF's default settings
# only one process will capture all VRAM on the GPU 
# see if anything is running on the GPU  
$ nvidia-smi  
# should show the running processes, and how much VRAM each of them takes
# you can also use top to monitor RAM and CPU
$ top
# since it's a development instance, it is ok to kill all python processes with pkill -9 python
# be carefull as it kills all the python processes that other people are running 
# it's ok to do it on our instance since it's consired to be only development zone for debugging
$ pkill -9 python
$ python av4_main.py &
$ exit
```
The network training may take hours, or days depending on your dataset and architecture of the network. It's important to note that in our code the epoch is counted by protein-ligand pairs, not by images. Every protein-ligand pair may have multiple incorrect positions of the ligand 50-400, and a single correct, crystal position. In this case, it takes 100 epochs to only show all of the negatives to the network once. That is different from classical understanding of epochs in image recognition when images can't have multiple frames.
Running the code should have resulted in four folders with outputs:
```
1_logs   
1_netstate   
1_test   
1_train  
```
`1_logs` might be empty, and will be used to write outputs during evaluation.
`1_netstate` will store the saved weights and biases for every trainable variable of the network 
(and also all other variables, such as epoch counter)
`1_train` and `1_test` should store summaries for variable states during training and testing that can
be visualized. Let's expect the outputs of in the foders

```
# log into our instance
$ ssh -i P2_key.pem ubuntu@awsinstance.com
# now I am
# ubuntu@ip-172-31-4-5:~$
# cd maksym
$ cd /core/summaries
$ cd 1_netstate
$ ls -l
# should show all of the files together with their size
# IE: 96789276 Jan 29 16:55 saved_state-60999.data-00000-of-00001
$ cd ../1_train
$ ls
# should show 
# events.out.tfevents.1485708632.ip-172-31-4-5
# which is a tensorflow summaries file
# let's try to visualize it:
# load tensorflow 0.12 (default version in the environment is 0.10)
$ source $TF12
# it's important to launch the tensorboard on port 80. By default internet browsers, such as chrome,
# will connect to port 80. You can read more here: 
# https://en.wikipedia.org/wiki/Port_(computer_networking)
# by default port 80 is not available to the user (the error is port is busy) that's why we use sudo
$ sudo python -m tensorflow.tensorboard --logdir=. --port=80
# now you can navigate your browser to awsinstance.com
```
 you should be able to see the following:
![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/cross_entropy.png)
Cross entropy (our cost function) goes down as we are training the network. 
![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/sparsity.png)
Sparsity of Rectifier Linear Unit is a percentage of zero-valued outputs of the layer. 
In chain rule for backpropagation, the derivative on sparse neuron is 0, and the derivative on downstream 
neurons is also 0. If the sparsity for the layer is exactly 1, backpropagation does not work, and weights 
can't be updated. That is what frequently happens when the network "explodes" because of the incorrect weight initialization.
![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/histogram.png)
Biases that all vere all initialized at 0.001 diverge as we are training our network. 


####Step 2: evaluating the network

In addition to four folders resulting from our previous step, you will need these three scripts:
```
1_logs   
1_netstate   
1_test   
1_train  

av4_eval.py
av4_input.py
av4_utils.py
```
the only new script is `av4_eval`  
```
av4_eval.py
# av4_eval script is very similar to av4_main
#
# crucial part 1:
# assembles all of the parts together:
# 1. input image creation pipeline              IE: ....something = image_and_label_queue 
# 2. network that makes predictions             IE: intuit_net or nilai_net
# 3. softmax instead of cost function in main   IE: tf.nn.softmax
# 4. no optimizer, variables are loaded         IE: saver.restore
# 5. run while loop pipe the tensor graph       IE: sess.run(train_step_run)
#
# crucial part 2: class store_predictions
# stores, sorts, and saves predictions
# estimates different evaluation parameters for straightforward binary classification such as 
# Area Under Curve https://en.wikipedia.org/wiki/Receiver_operating_characteristic
# Confusion Matrix https://en.wikipedia.org/wiki/Confusion_matrix
#  
# crucial part 3: av4_eval can be used for two different tasks
# 1. distinguishing the correct position within many positions as in docking
# 2. sorting ligands each of which has many positions as in sorting
# 1 is very straightforward since our training consits of correct and incorrect positions, 
# we only need to score all of the available positions
# our performance on task 1 is very high (AUC > 0.94)
# 2 is not straighforward. Since many of the docked positions given to the network are not correct
# (sometimes all of them)
```
Now let's evaluate our script on distinguishing a single correct position from a single incorrect position, the same task it has been trained on. In this case testing set would be the part of the same dataset that was not used for training.

```
# Let's download the dataset from Kaggle to our local machine
# navigate your browser to: https://inclass.kaggle.com/c/affinity4/data
# and download holdout_av4.zip
$ scp -i P2_key.pem holdout_av4.zip ubuntu@awsinstance.com:/home/ubuntu/common/data
$ ssh -i P2_key.pem ubuntu@awsinstance.com
$ cd common
# unzip the database 
$ unzip holdout_av4.zip
# get the path to current directory
$ pwd 
# /home/ubuntu/common/data/labeled_av4
$ cd ~/maksym/core/summaries/1_netstate
$ ls
# note the latest step of the saved network
# it's saved_state-60999.data-00000-of-00001 in my case
$ cd ../..
# edit 
# FLAGS.saved_session = ./summaries/1_netstate/saved_state-60999
# FLAGS.database_path = /home/ubuntu/common/data/labeled_av4
$ vi av4_eval.py
# now source tensorflow 0.12 and launch the evaluation script
$ source $TF12
$ python av4_eval.py
# ....
# current_epoch: 6 batch_num: [82]  prediction averages: 0.538309   examples per second: 273.89
# ......
# all_done
# the evaluation script should have written five files into the corresponding logs folder
# in our case it's 
# saved_state-60999_average_submission.csv
# saved_state-60999_max_submission.csv
# saved_state-60999_multiframe_submission.csv
# saved_state-60999_predictions.txt
# saved_state-60999_scores.txt
# for this kind of evaluation only two files are meaningful:
$ vi saved_state-60999_predictions.txt
# saved_state-60999_predictions.txt
# has four columns:
# average_prediction   label   filename   predictions
#
# 1.0       1.0       1swd_465_ligand.av4_frame19                       1.0           
# 1.0       1.0       2pno_1757_ligand.av4_frame11                      1.0        
# 1.0       1.0       4d1j_4337_ligand.av4_frame13                      1.0             
# 1.0       1.0       4nul_138_ligand.av4_frame11                       1.0         
# 1.0       1.0       2pno_1757_ligand.av4_frame3                       1.0           
# 1.0       1.0       4nul_138_ligand.av4_frame15                       1.0,1.0            
# 1.0       1.0       4nul_138_ligand.av4_frame17                       1.0,1.0   
# .....
# ...
# 0.953     1.0       3n66_819_ligand.av4_frame9                        0.953    
# 0.953     1.0       3elz_401_ligand.av4_frame5                        0.953      
# 0.953     1.0       4rrw_2217_ligand.av4_frame8                       0.953 
# 0.953     1.0       1gt6_538_ligand.av4_frame6                        0.953  
# 0.953     1.0       1an5_533_ligand.av4_frame18                       0.953 
# 0.953     1.0       4ki0_1898_ligand.av4_frame2                       0.953 
# 0.953     1.0       1ivf_807_ligand.av4_frame18                       0.953  
# ....
# 0.002     0.0       3ekw_199_ligand.av4_frame3                        0.002   
# 0.002     0.0       2b0m_362_ligand.av4_frame0                        0.003,0.001   
# 0.002     0.0       1oya_399_ligand.av4_frame14                       0.002 
# 0.002     0.0       2nxi_3110_ligand.av4_frame3                       0.002   
# 0.002     0.0       1yrh_1605_ligand.av4_frame5                       0.002   
# 0.001     0.0       3thq_430_ligand.av4_frame18                       0.001 
# 0.001     0.0       1jvu_248_ligand.av4_frame2                        0.001
# 0.001     0.0       1yrh_1605_ligand.av4_frame17                      0.001
#
# the reson that the last column has multiple entries is because same protein-ligand complex can be
# evaluated several times. Because random affine transform (in av4_input) rotates and shifts the box 
# around protein-ligand complex randomly, every time an image in different orientation is evaluated
# ideally, the network should be rotationally and translationally invariant. In that case all of the
# values in the last column should be same. That is almost the case.
# another file to look at is saved_state-60999_scores.txt
# some of the importat parameters such as AUC
# AUC: 0.947862873006
# and the confusion matrix: [[7835 1141] [1019 7764]]
# should be here.
# you can read more about AUC and the confusion matrix here: 
# https://en.wikipedia.org/wiki/Receiver_operating_characteristic
```
We have applied our network to distinguish correct position of ligand from incorrect (docking), and it performed very well. Now let's try to apply our network to another stage of virtual screening - ranking. In this case we have multiple ligands (flexible keys), and a series of proteins (rigid locks). In this case we will not do the docking itself, but will use several proposed positions (400) by [smina](https://github.com/mwojcikowski/smina) for each of the ligands. 
`unlabeleled_av4` contains 10 receptors and, on average, 200 ligands per receptor (100 actives and 100 inactives). There are top 400 positions predicted by smina positions in each ligand av4 file.
```
# edit the name of the database to be used for evaluations
# to the location of the database at: /home/ubuntu/common/data/labeled_av4
$ vi av4_eval.py
# run the eval script
$ python av4_eval.py
# the number of epochs in the script FLAGS.num_epochs 
# will determine the number of frames per each ligand to be evaluated
# this time we will re-rank only top 20 positions and not consider other 380
# inspect the outputs of the script at 1_logs
# cd ./summaries/1_logs
# again you may find five files in the same folder:
$ ls
# saved_state-60999_average_submission.csv
# saved_state-60999_max_submission.csv
# saved_state-60999_multiframe_submission.csv
# saved_state-60999_predictions.txt
# saved_state-60999_scores.txt
#
# this time another three files will carry meaning:
$ less saved_state-60999_average_submission.csv
# ID,Predicted
# 3zw2_270_ligand,0.944675922394
# 1rmg_432_ligand,0.979166805744
# 1b30_303_ligand,0.954633176327
# 3mw7_516_ligand,0.999969124794
# 3pby_439_ligand,0.999946951866
# 4mm5_499_ligand,0.999761760235
# 1ppf_286_ligand,0.954079449177
# ...
# ..
# saved_state-60999_average_submission.csv
# will store frame averages
# saved_state-60999_max_submission.csv
# will store maximum of the frames
# and can be submitted to Kaggle directly
# saved_state-60999_multiframe_submission.csv
# will store all of the predictions separated by comma an can be used for future analysis
# 
# move all of the predictions to local machine
$ cd ..
$ tar zcvf 1_logs.tar.gz 1_logs
$ pwd
# /home/ubuntu/maksym/core/summaries
$ exit
$ scp -i P2_key.pem ubuntu@awsinstance.com:/home/ubuntu/maksym/core/summaries/1_logs.tar.gz .
$ tar -xzvf 1_logs.tar.gz
$ cd 1_logs
# now you are ready to submit your solution:
# please, navigate your browser to inclass/kaggle.com/c/affinity4
# how much did you score
# you can also use post-process 
# saved_state-60999_multiframe_submission.csv
# for example, auc_script.py under /misc/ can calculate AUC for any given protein
# receptor: aa2ar
# 0.575214145789
# num predictions: 365
# receptor: cp2c9
# 0.540473188014
# num predictions: 627
# ...
# ..
```


####Step 3: database preparation (demo)
_We are working hard to make our database construction scripts human-readable. We hope to finish in the near future_  

Database construction is a very complex multistage process, perhaps much more complex than Affinity itself. There are few important things to notice.
Database is constructed from X-ray crystallographic images from the [Protein Data Bank](http://www.rcsb.org/).
And binding affinity data is taken from databases like [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
In the process of preparation ligand is split from it's protein. In addition, fake positions can be generated with standard virtual screening algorithms like [Vina](http://vina.scripps.edu/manual.html).  
On the final stage `av4_database_master` joins pdb recods into a single database of binary files that can be read very fast during training of the network.
Data and .av4 format is, generally, stored in the following way
```
# [label_for_frame_1
#  [[x11,y11,z11]
#   [x12,y12,z12]
#   [x13,y13,z13]
#   [x14,y14,z14]]
#   [atom_tag11,atom_tag12,atom_tag13,atom_tag14]
# [label_for_frame_2
#  [[x21,y21,z21]
#   [x22,y22,z22]
#   [x23,y23,z23]
#   [x24,y24,z24]]
#   [atom_tag21,atom_tag22,atom_tag23,atom_tag24]
# ..........

```
Finally, the naming convention of the database is the following: 
```1a28
     1a28.av4
     1a28_500_ligand.av4
     1a28_501_ligand.av4
 ```
 [1a28](http://www.rcsb.org/pdb/explore.do?structureId=1a28) is the structure ID in the PDB. `1a28.av4` is the protein itself, and `1a28_500_ligand.av4` and `1a28_501_ligand.av4` are two of it's ligands.
 
####Step 4: running affinity on Bridges, XSEDE national supercomputer
In steps 2 and 3 we have used Amazon instance that only has a single GPU to run the scripts. It make take up to a few days to train a deep CNN on protein images, and in industrial and scientific applications it's very reasonable to train networks with several different architectures at the same time. XSEDE (Extream Science and Engineering Discovery Environment) is a broad effort that controls access to most of the largest clusters in the US. In this tutorial we will use [Bridges](https://portal.xsede.org/psc-bridges) that has over a 100 GPUs. The current version of Affinity is written for a single GPU. But usually every user submits a few separate, unrelated jobs at the same time. 

```
# register for XSEDE here: https://portal.xsede.org/#/guest
# and ask maksym to add you to our computer time grant 
#
# login to Bridges through XSEDE Single Sign-On (SSO) Hub. 
bash
$ ssh [xsede_username]@login.xsede.org
$ gsissh bridges
# each user is prowided with a high performance work directory would be different from your $HOME 
# (IE: /home/korablyo) for me. Work directory allows to read and write data much faster 
# you will need to find your work directory and put the database there
# it is: /pylon5/[groupname]/[username]
# get groupname
bash
$ id -gn
# clone Affinity source code to your work directory
bash
$ cd /pylon5/[groupname]/[username]
$ git clone https://github.com/mitaffinity/core.git
# now you will need to log in into our AWS instance to download the database to Bridges
# you will need a .pem access key 
# just create a new text file , and copy-paste your key there
vi P2_key.pem
# paste
:wq
# change permissions of the key so that only you can read it 
# you can read more here: http://stackoverflow.com/questions/9270734/ssh-permissions-are-too-open-error
$ cd $HOME
$ chmod 400 key.pem
# transfer data from aws instance to bridges
bash
$ cd  /pylon5/[groupname]/[username]
$ scp -i $HOME/key.pem ubuntu@awsinstance.com:/home/ubuntu/common/data/labeled_av4.zip ./
$ unzip labeled_av4.zip
# change FLAGS.database_path in av4_main.py
# to database_path = "/pylon5/[groupname]/[username]/labeled_av4"

```
create batch script to submit your job. You can read more about submission queue [here](https://www.psc.edu/index.php/bridges/user-guide/running-jobs).
In this case it is just a test file that will be executed by bash. You can use `vi` or any other command line text editor to create it.
```
#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --ntasks-per-node 28
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:k80:4
#echo commands to stdout
set -x

#load module
module load cuda/8.0
#module load tensorflow/0.12.1
# TF is compiled with 1.1 as of April 15th 2015
# as og apr 15th, Tensorflow 1.1 is not available yet on Bridges.
# needs fix
module load tensorflow/0.9.0

#set python environment
source $TENSORFLOW_ENV/bin/activate

#move to working directory
cd /pylon5/[groupname]/[username]/core

#run GPU program
python av4_main.py
```
 run your job
```
$ sbatch job.sh
# monitor the status of your job with 
$ squeue -u korablyo

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
[korablyo@br006 22_single_fastnet_no_H_repeat2]$ squeue -u korablyo
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            903126       GPU  gpu4.sh korablyo PD       0:00      1 (Resources)
            903137       GPU gpu5m.sh korablyo PD       0:00      1 (Resources)
            903157       GPU gpu5t.sh korablyo PD       0:00      1 (Resources)
            903158       GPU  gpu6.sh korablyo PD       0:00      1 (Resources)
            903170       GPU gpu16.sh korablyo PD       0:00      1 (Resources)
            903173       GPU gpu17.sh korablyo PD       0:00      1 (Resources)
            903176       GPU gpu18.sh korablyo PD       0:00      1 (Resources)
            903370       GPU gpu13.sh korablyo PD       0:00      1 (Resources)
            903375       GPU gpu14.sh korablyo PD       0:00      1 (Resources)
            903377       GPU gpu21.sh korablyo PD       0:00      1 (Resources)
            903379       GPU gpu22.sh korablyo PD       0:00      1 (Resources)
            899255       GPU   gpu.sh korablyo  R    8:37:34      1 gpu038
            900227       GPU   gpu.sh korablyo  R    8:26:57      1 gpu021
            899232       GPU   gpu.sh korablyo  R 1-16:53:48      1 gpu043
            899231       GPU   gpu.sh korablyo  R 1-16:57:49      1 gpu034
            899229       GPU   gpu.sh korablyo  R 1-19:32:28      1 gpu040
            899227       GPU   gpu.sh korablyo  R 1-21:57:01      1 gpu041
            901652       GPU   gpu.sh korablyo  R    7:37:56      1 gpu019
            901668       GPU   gpu.sh korablyo  R    7:37:56      1 gpu026
            899233       GPU   gpu.sh korablyo  R 1-04:00:11      1 gpu042
            903120       GPU  gpu1.sh korablyo  R    2:04:47      1 gpu015


# shows that I have 10 jobs running and 11 jobs pending

```

### Running job on xstream
login to Bridges through XSEDE Single Sign-On (SSO) Hub.
```bash
ssh [xsede_username]@login.xsede.org
```
login xstream
```bash
gsissh xstream
```
work directory
```bash
cd $WORK
```

### example slurm script
```bash
#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res_%j.txt
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000
#SBATCH --gres gpu:2
#SBATCH --gres-flags=enforce-binding

module load foss/2015.05
module load CUDA/7.5.18
module load cuDNN/5.1-CUDA-7.5.18
module load tensorflow/1.0.0

srun  python jobs.py
```
