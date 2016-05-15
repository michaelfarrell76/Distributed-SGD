# Distributed-SGD for Python
An implementation of distributed stochastic gradient descent in python. Clients can be local and remote. For this task, you can download the data from http://www.vision.caltech.edu/Image_Datasets/Caltech101/.

## Requirements

This code is written entirely in Python, and an installation of gRPC, Numpy, Scipy, and Autograd are necessary. These packages can be easily installed through PIP using the following commands. 

```bash
$ pip install numpy
$ pip install scipy
$ pip install autograd
$ pip install grpcio
```

For launching the code remotely, we will be working with Google Cloud Compute. In order to interact with GCloud instances, please install the GCloud sdk. This is located here: https://cloud.google.com/sdk/.  

## Directory Table of Contents
```
.
├── 101_ObjectCategories		    # Folder holding the raw data from the 101_ObjectCategories
|-- data                            # Folder holding the processed data
├── client.py                       # Python script used to initiate a client
|── server.py 						# Python script to manually initiate a server
├── dist_sgd_pb2.py                 # Automatically compiled protobufs for the parameter server
├── README.md                       # Python usage 
├── images(16).npy                  # Extremely small dataset included for reference
├── output_labels(16).npy           # Classifications of each image for the extremely small dataset
├── nnet           					# Folder that includes a module for a convolution neural net
├── protobuf_utils         			# Folder that includes utilities for manipulating tensor protobuffers
├── run_codegen.sh                  # Shell command used to generates the protobuffers 
└── start.sh                        # Script that launches client.py on when running within gCloud
```

## Description

## Local Usage Instructions
To launch clients locally, in three different terminals, simply run:
```bash
$ python client.py --id 1 
$ python client.py --id 2
$ python client.py --id 3
```

#### Remote Usage Instructions

##### Create a baseline startup image

We only have to setup and install everything once, after which we can clone that image repeatedly when we launch VMs. 

###### Create the image
- Click on the 'VM Instances' tab
- Create Instance
- Give the instance a name i.e. 'train-conv-nn'
- Set the zone to us-central1-b
- Choose 2vCPU highmem as machine type
- Under boot disk click change
- Choose Ubuntu 14.04 LTS
- At the bottom change size to 30 GB and click 'select'
- Allow HTTP traffic
- Allow HTTPS traffic
- Click 'Management, disk, networking, SSH keys' to dropdown more options
- Under 'Disk' unclick 'Delete boot disk when instance is deleted'
- Click 'Create' an you should see your new instance listed in the table

###### Setup the disk
- Run the command gcloud init and log into your Google Cloud account
- Run the command to SSH into your instance:
```bash
$ gcloud compute ssh train-conv-nn --zone us-central1-b
```
- After logging in, we can clone the repository and install the necessary requirements.
- Once the server is setup to your liking, disconnect from the server and return to your google cloud dashboard
- Go to the 'VM Dashboard'
- Click on the instance you just setup, and delete it. This should remove the instance and save it as a disk. If you click on the 'disks' tab, you should see the instance name you just deleted.

###### Create the image

- Click on the 'Images' tab
- 'Create Image'
- Give it a name i.e. 'train-conv-image'
- Under Source-Disk, choose the disk that you just created 
- Create

##### Generate an 'Instance Template'
- Click on the 'Instance templates' tab
- Create new
- Name the template i.e. 'train-conv-template'
- Under 'Boot Disk' click change
- At the top click 'Your image'
- Choose the image you just created i.e. 'train-conv-image'
- Set size to 30 GB
- Select
- Allow HTTP traffic
- Allow HTTPS traffic
- Under more->Management, include cd ~/distributed-sgd/python-python; sh start.sh
  in startup script
- Under more->Disks, unclick 'Delete boot disk when instance is deleted'
- Create

##### Generate an 'Instance Group'
- Go to the "Instance groups" tab
- Create instance group
- Give the group a name, i.e. 'train-conv-group'
- Give a description
- Set zone to us-central1-b
- Use instance template
- Choose the template you just made i.e. 'train-conv-template' 
- Set the number of instances
- Create
- Wait for the instances to launch
- Once there is a green checkmark, click on the new instance

All instances in the instance group are now running the python client.py command and will begin training.
SSH into any of the instances to see their progress.

## Acknowledgments

Our implementation adapts code for the convolutional neural net from the Autograd convolution neural net example:

* [Autograd](https://github.com/HIPS/autograd)