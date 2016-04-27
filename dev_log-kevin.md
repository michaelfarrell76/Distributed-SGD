
Played around with autograd in python. Looking for a reasonable toy dataset to test sgd on distributed system
Looked into the convolutional network example for autograd https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
This ended up being perfect because it spits out a long vector of gradients that it uses 
Looking for a far heavier dataset. MSINT runs in a 1-2 minutes.
Found Caltech 101, built some preprocessing code, modified some of the code for the neural network
Needed to downsize the images substantially. 240 x 240 is around 12 GB of data. Shrunk it down to 128 x 128, making it 4 Gb of data. New gradients are around 0.5Gb. This makes network speeds pretty prohibitive though. 
Epochs take a couple minutes to run. Batches takes around 10-15 seconds each. Seems rather reasonable


Looking into Azure for launching VMs 
Discovered CLI for Azure
Set up 5 different accounts all using the Bizspark subscription. One email account also has a free subscription activated.
Emails and passwords are listed below:

(candokevin2@hotmail.com, cs262michaelkevin)
(candokevin3@hotmail.com, cs262michaelkevin)


Log into portal.azure.com to interact more with the system

Received instructions from Mike on how to setup grpc. For replicability on later Linux VMs we launch, I've documented the steps
I took below:

Set up Protobufs 3.0.0
	https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-python-3.0.0-beta-2.zip
	./autogen.sh
	./configure
	make
	make check
	make install

Set up grpc
	git clone https://github.com/grpc/grpc.git
	sudo make grpc_python_plugin
	sudo vim /etc/paths, add the line /Users/candokevin/stash/grpc/bins/opt
	

It might be a good idea to look into Docker containers, and Docker networks for launching and setting up VMs. 