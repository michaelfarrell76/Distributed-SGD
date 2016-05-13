#!/bin/bash
# 
# gcloud_startup.sh
#	
# This is a bash script that is used to setup a google cloud server. This script
#	will install the following on the server:
#		- git
#		- luarocks
#		- pip
#		- torch
#		- lua-parallel (local version)
#		- rnn (torch) 
#		- hdf5 (torch)
#		- anaconda
#		- h5py
# The script will also clone the Distributed-SGD repo onto the server

# Ensure that git is installed
if hash git &> /dev/null
then
	echo -e "\033[0;32mgit installed\033[0m"
else
	echo -e "\033[0;34mInstalling git ...\033[0m"
	(echo "Y" | sudo apt-get install git) > /dev/null  
fi

# Ensure that luarocks is installed
if hash luarocks &> /dev/null
then
	echo -e "\033[0;32mluarocks installed\033[0m"
else
	echo -e "\033[0;34mInstalling luarocks ...\033[0m"
	(echo "Y" | sudo apt-get install luarocks) &> /dev/null  
fi

# Ensure that pip is installed
if hash pip &> /dev/null
then
	echo -e "\033[0;32mpython-pip installed\033[0m"
else
	echo -e "\033[0;34mInstalling python-pip ...\033[0m"
	(echo "Y" | sudo apt-get install python-pip) > /dev/null  
fi

source ~/.profile

# Ensure that torch is installed
if hash th &> /dev/null
then
	echo -e "\033[0;32mtorch installed\033[0m"
else
	echo -e "\033[0;34mInstalling torch ...\033[0m"
	git clone https://github.com/torch/distro.git ~/torch --recursive &> /dev/null
	cd ~/torch
	bash install-deps 2&>1 > /dev/null 
	echo "yes" | ./install.sh 2&>1 > /dev/null 
	cd ..
	source ~/.profile
fi

# Ensure that rnn is installed
if (luarocks list | grep -q rnn) &> /dev/null  
then
	echo -e "\033[0;32mrnn installed\033[0m"
else
	echo -e "\033[0;34mInstalling rnn ...\033[0m"
	luarocks install rnn &> /dev/null  
fi

# Ensure that torch-hdf5 is installed
if (luarocks list | grep -q hdf5) &> /dev/null  
then
	echo -e "\033[0;32mhdf5 installed\033[0m"
else
	echo -e "\033[0;34mInstalling hdf5 ...\033[0m"
 	echo "Y" | sudo apt-get install libhdf5-serial-dev hdf5-tools > /dev/null  
 	git clone https://github.com/deepmind/torch-hdf5.git &> /dev/null
 	cd torch-hdf5
 	luarocks make hdf5-0-0.rockspec &> /dev/null  
 	cd ..
fi

# Make sure that the Distributed SGD is downloaded and isntalled
if [ -e "Distributed-SGD" ]
then 
	# Update the repos
	echo -e "\033[0;34mPulling Distributed-SGD repo changes ...\033[0m"
	cd Distributed-SGD
	git pull &> /dev/null
	cd lua-lua/End-To-End-Generative-Dialogue
	echo -e "\033[0;34mPulling End-To-End-Generative-Dialogue repo changes ...\033[0m"
	git pull origin master &> /dev/null

	cd ../../..
else
	# Clone repo and install parallel
	echo -e "\033[0;34mCloning repo Distributed-SGD ...\033[0m"
 	git clone --recursive https://github.com/michaelfarrell76/Distributed-SGD.git &> /dev/null
 	cd Distributed-SGD/lua-lua
 	bash install_parallel.sh 
 	cd ../../
fi

# Ensure that anaconda is installed 
if [ -e "anaconda2" ]
then 
	echo -e "\033[0;32manaconda installed\033[0m"
	echo -e "\033[0;34mInstalling h5py ...\033[0m"

	# Install hdf5 for python
	echo "y" | conda install h5py &> /dev/null
else
	echo -e "\033[0;34mDownloading anaconda ...\033[0m"
	wget http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh &> /dev/null
	echo -e "\033[0;34mInstalling anaconda ...\033[0m"
	bash Anaconda2-4.0.0-Linux-x86_64.sh -b > /dev/null
	rm Anaconda2-4.0.0-Linux-x86_64.sh
	echo 'export PATH="/home/michaelfarrell/anaconda2/bin:$PATH"' > .bashrc
	echo -e "\033[0;33mIn order for python to be run, you must logout and log back in\033[0m" 
fi

