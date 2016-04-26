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
