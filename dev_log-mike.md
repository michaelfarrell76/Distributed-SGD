- need to install proto3 protocol buffers

download link:
https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-python-3.0.0-beta-2.zip

https://github.com/google/protobuf

example:
https://github.com/grpc/grpc/tree/release-0_13/examples/python/helloworld

cd into directory
	brew update && brew remove gmp && brew install gmp && brew link gmp

	./autogen.sh

	./configure

	make

	make check

	make install

example usage
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto

- installed grpc according to the following instructions listed here: https://github.com/grpc/grpc/tree/release-0_13/examples/python an outline of the command I ran are the following:

	sudo pip install grpcio

git clone https://github.com/grpc/grpc

	- We can test to see if the helloworld example works:

	cd grpc/examples/python/helloworld

	- Run the server

	python2.7 greeter_server.py &

	- Run the client

	python2.7 greeter_client.py

	-You should see the output "Greeter client received: Hello, you!"

Instead going to copy the necessary files into our directory and have a small running example

in the folder Distributed-SGD/helloworld:

have the files:

	greeter_client.py
	greeter_server.py 


sudo pip install grpcio --upgrade





HOW I GOT IT TO WORK
Used this link:
https://github.com/grpc/homebrew-grpc


curl -fsSL https://goo.gl/getgrpc | bash -

 virtualenv venv
source venv/bin/activate

curl -fsSL https://goo.gl/getgrpc | bash -s python

cd venv

git clone https://github.com/grpc/grpc.git

cd grpc

make grpc_python_plugin




here we go:

cd /usr/local/
mkdir manual
cd manual

curl -fsSL https://goo.gl/getgrpc | bash -

virtualenv venv

source venv/bin/activate

curl -fsSL https://goo.gl/getgrpc | bash -s python

pip install numpy
pip install scipy
sudo pip install pillow
pip install sklearn
pip install autograd

cd venv

git clone https://github.com/grpc/grpc.git
cd grpc

make grpc_python_plugin

sudo vim /etc/paths

	and add the line:

	/usr/local/manual/venv/grpc/bins/opt



BEFORE RUNNING ANYTHING

source /usr/local/manual/venv/bin/activate


Important links:
https://github.com/grpc/homebrew-grpc
https://docs.docker.com/engine/userguide/networking/
http://www.bpython-interpreter.org
https://github.com/mila-udem/fuel






