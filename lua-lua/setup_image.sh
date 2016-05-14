#!/usr/bin/env bash
# 
# setup_image.sh
#	
# This is a bash script that is used to setup an image on the google cloud server
#	it copies over the startup script, runs the script, disconnects and reconnects,
#	then reruns the startup script

# Copy over the startup script
scp -o "StrictHostKeyChecking no" -i ~/.ssh/dist-sgd-sshkey startup.sh $USERNAME@$EXTERNAL_IP:~/

# Run the startup script on the server
echo "bash startup.sh" | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/dist-sgd-sshkey $USERNAME@$EXTERNAL_IP

# Disconnect from the server, reconnect  and finish running last things needed for initialization
echo "bash startup.sh; " | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/dist-sgd-sshkey $USERNAME@$EXTERNAL_IP
