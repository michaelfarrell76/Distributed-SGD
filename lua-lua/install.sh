#!/usr/bin/env bash
# 
# Install script for lua-lua Distributed SGD System
# 
# The dependencies installed are:
#	- local version of parallel
# 	- 


# Move into install directory
cd ..
if [ -e "install" ]	
then
	echo -e "\033[0;32minstall folder exists\033[0m"
else
	echo -e "\033[0;34mMaking install repo ...\033[0m"
	mkdir install
fi

# Ensure that parallel is downloaded and installed with local version
if [ -e "lua---parallel" ]	
then
	echo -e "\033[0;32mparallel installed\033[0m"
else
	echo -e "\033[0;34mCloining Parallel Repo ...\033[0m"
	git clone https://github.com/clementfarabet/lua---parallel.git
	cd lua---parallel
	echo -e "\033[0;34mCopying local init.lua file for parallel...\033[0m"
	cp ../../lua-lua/parallel/init .
	echo -e "\033[0;34mBuilding local version of parallel...\033[0m"
	luarocks make
fi

echo -e "\033[0;32mInstall complete\033[0m"

