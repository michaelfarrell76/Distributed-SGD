# Distributed-SGD: lua-lua
An implementation of distributed stochastic gradient descent in lua/torch. Clients can be local and remote. 

## Requirements

This code is written in Lua, and an installation of [Torch](https://github.com/torch/torch7/) is assumed. Training requires a few packages which can easily be installed through [LuaRocks](https://github.com/keplerproject/luarocks) (which comes with a Torch installation). Datasets are formatted and loaded using [hdf5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format), which can be installed using this [guide](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md).

Once torch and torch-hdf5 are installed, use luarocks to install the other dependencies used in the example:

```bash
$ luarocks install nn
$ luarocks install rnn
```
If you want to train on an Nvidia GPU using CUDA, you'll need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) as well as the `cutorch` and `cunn` packages:
```bash
$ luarocks install cutorch
$ luarocks install cunn
```
We need to ensure that our local version of parallel is installed. This can be done with a short bash script from the lua-lua folder:
```bash
$ cd lua-lua
$ ./install_parallel.sh
```

## Directory Table of Contents
```
.
├── data     # Folder holding data used for demo
├── parallel # Folder containing the changes we added to the parallel class
├── End-To-End-Generative-Dialgoue # Folder of our other repo containing the code used in demo
├── README.md # lua-lua usage 
├── server.lua # Main server file
├── README.md
├── install_parallel.sh # script that installs our version of parallel
└── demo_server.lua # A demo class that implements the server
```

## Description

## Demo-Usage
Code is run from the lua-lua folder:
```bash
$ cd lua-lua
```

#### Local

to run a worker with 4 parallel clients on your own machine:
```bash
$ th server.lua -n_proc 4
```

#### Remote - localhost

In order to get the demo to connect through localhost rather than simply forking, we must first setup an .ssh key for this project. 

Note: This is basically doing the same thing as [local](https://github.com/michaelfarrell76/Distributed-SGD/blob/master/lua-lua/README.md#local), except we now connect to the clients through localhost. This is a good tool to use to debug problems with clients running on remote servers.

###### Generate ssh key
Replace USERNAME with your username on the computer you want to connect to (i.e., USERNAME = michaelfarrell).
```bash
$ USERNAME=michaelfarrell
$ ssh-keygen -t rsa -f ~/.ssh/dist-sgd-sshkey -C $USERNAME
```
Hit enter twice and a key should have been generated. 

###### Add ssh-key to authorized keys

In order to connect to clients through localhost, we must add the key to our list of authorized_keys:
```bash
$ cat ~/.ssh/dist-sgd-sshkey.pub >> ~/.ssh/authorized_keys
$ chmod og-wx ~/.ssh/authorized_keys 
```

###### Connect via localhost

You can now communicate over localhost using the command:

```bash
$ th server.lua -n_proc 4 -localhost
```

#### Remote - gcloud 

Instead of having the client programs running on your own computer, you can farm them out to any number of remote computers. Below is a description of how to setup remote clients using google cloud. 

###### Adding ssh key to gcloud servers

If you have yet to do so, [generate an ssh-key](https://github.com/michaelfarrell76/Distributed-SGD/blob/master/lua-lua/README.md#generate-ssh-key)

Once you have created the key print it out:

```bash
$ cat ~/.ssh/dist-sgd-sshkey.pub
```

Next you must add the key to the set of public keys :
- Login to our google compute account. 
- Go to compute engine dashboard
- Go to metdata tab
- Go to ssh-key subtab
- Click edit
- Add the key you copied as a new line

Restrict external access to the key:
```bash
$ chmod 400 ~/.ssh/dist-sgd-sshkey
```

###### Generate an 'Instance Template'
- Click on the 'Instance templates' tab
- Create new
- Name the template 
- Choose 8vCPU highmem as machine type
- Choose Ubuntu 14.04 LTS as boot disk
- Allow HTTP traffic
- Allow HTTPS traffic
- Under more->Disks, unclick 'Delete boot disk when instance is deleted'
- Create

###### Allow tcp connections
- Click on the 'Instance templates' tab
- Click on the new template you created
- Go down to networks and click on the 'default' link
- Go to 'Firewall rules' and Add a new rule
- Set name to be 'all'
- Set source filter to allow from any source
- Under allowed protocols, put 'tcp:0-65535; udp:0-65535; icmp'
- Create


###### Generate an 'Instance Group'
- Go to the "Instance groups" tab
- Create instance group
- Give the group a name, i.e. training-group-dev
- Give a description
- Set zone to us-central1-b
- Use instance template
- Choose 'miket=template' or other template of choice
- Set the number of instances
- Create
- Wait for the instances to launch
- Once there is a green checkmark, click on the new instance

###### Connecting to gcloud servers

You can connect to one of the servers by running:
```bash
$ IP_ADDR=130.211.160.115
$ ssh -o "StrictHostKeyChecking no" -i ~/.ssh/dist-sgd-sshkey $USERNAME@$IP_ADDR
```
where $username is the username you used to create the ssh key as defined above, and IP_ADDR is the ip address of the machine listed under "External ip" (i.e., 104.197.9.84). Note: the flag `-o "StrictHostKeyChecking no"` automatically adds the host to your list and does not prompt confirmation.

If you get an error like this:
```bash
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```
then you'll want to
```bash
$ vim ~/.ssh/known_hosts
```
and delete the last few lines that were added. They should look like some ip address and then something that starts with AAAA. You can delete lines in vim by typing 'dd' to delete the current line. This can happen when you restart the servers and they change ip addresses, among other things.

###### Adding remote clients
You will want to add your list of client servers to the file 'client_list.txt' where each line in the file is one of the external ip addresses located in the Instance group you are currently using. 

###### Initializing remote servers
Before using the remote servers, we need to make sure that the servers are ready to go. This can be done by running
```
$ python server_init.py
```
from the src folder on your own computer. 

###### Running the remote server:
If the servers have been initialized, you will first want to connect to one of them:
```bash
$ ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey $USERNAME@$IP_ADDR
```

###### Running code with remote clients

Once connected, you need to again setup an ssh key from the computer that you are using as the client.

Again:

1) [generate an ssh-key](https://github.com/michaelfarrell76/Distributed-SGD/blob/master/lua-lua/README.md#generate-ssh-key)
2) [add key to gcloud server account](https://github.com/michaelfarrell76/Distributed-SGD/blob/master/lua-lua/README.md#adding-ssh-key-to-gcloud-servers)

Once this is done, you can run the server with remote gcloud clients using the command:
```bash
$ cd Distributed-SGD/lua-lua
$ th server.lua -n_proc 4 -remote

```

## For Personal Usage

If you wish to extend this demo to work with your own SGD model you must simply create a new server class specific to your task, replacing the 'demo_server' class. Use the file 'demo_server.lua' as an example. The server only needs to have __init(opt) and run() functions defined in order to work. Once this class is properly defined (i.e. named 'new_server'), you can run the following to initiate your task:

```bash
$ NEW_SERVER_NAME=new_server
$ th server.lua -server_class $NEW_SERVER_NAME # Plus Additional arguments 

``` 

When developing, all command line arguments should be added in the file server.lua. Look at the command arguments (th server.lua --help) that already exist and use those names when developing your model. If you need an additional command line argument, add it in server.lua. Other than this, there should be no reason to edit the server.lua file. 


## TODO
- Document data folder and include description in demo-usage about what the demo is
- Add in documentation of how the data needs to be formatted in order to run the demo
- Finish description
- Finish demo-usage 
- Clean up demo_server.lua
- Finish Acknowledgements
- Add in proto implementation

## Acknowledgments
This example is also apart of another one of our repos: https://github.com/michaelfarrell76/End-To-End-Generative-Dialogue

Our implementation utilizes code from the following:

* [Yoon Kim's seq2seq-attn repo](https://github.com/harvardnlp/seq2seq-attn)
* [Element rnn library](https://github.com/Element-Research/rnn)
* [Facebook's neural attention model](https://github.com/facebook/NAMAS)
