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
Lastly we need to ensure that our local version of parallel is installed. This can be done with a short bash script:
```bash
$ ./install_parallel.sh
```

## Usage

## Acknowledgments
This example is also apart of another one of our repos: https://github.com/michaelfarrell76/End-To-End-Generative-Dialogue

Our implementation utilizes code from the following:

* [Yoon Kim's seq2seq-attn repo](https://github.com/harvardnlp/seq2seq-attn)
* [Element rnn library](https://github.com/Element-Research/rnn)
* [Facebook's neural attention model](https://github.com/facebook/NAMAS)