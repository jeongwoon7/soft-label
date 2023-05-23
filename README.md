# Utilizing soft labels of Convolutional Neural Networks for predicting quantum transmission probability: An application to high-dimensional data prediction

by Moon-Hyun Cha and Jeongwoon Hwang, submitted

1. Data for ML models : test_lhc, test_normal, test_uniform, test_adaptive_git, sample_git
* Each of them are generated by Latin hypercube sampling, random sampling with normal distribution, random sapling with uniform distribution
* In each directory, there're figures (224x224 image data mimicking quantum dot chains) and corresponding labels (transmission probability data represented by 1000 data points)
* test_adaptive_git is the data for adaptive sampling
* sample_git is the data for Feed-forward NN

2. Code for machine learning : basic_run.py, adaptive_run.py, FNN_run.py
* In terminal, python3 basic_run.py > stdout

3. Module containing classes & functions used for the ML : Module.py
* Customdataset, ResNet, loss_pred_ResNet, train_loop, valid_loop
 
4. Code for analysis : Analysis.py
