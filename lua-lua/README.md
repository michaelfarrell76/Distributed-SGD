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
├── server.lua # Main server object
├── README.md
├── install_parallel.sh # script that installs our version of parallel
└── demo_server.lua # A demo class that implements the server
```

## Description

## Demo-Usage

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
