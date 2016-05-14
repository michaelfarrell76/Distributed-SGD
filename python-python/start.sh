#! /bin/bash
#! /home/candokevin/anaconda2/bin/python
cd /home/candokevin/stash/distributed-sgd/python-python
git pull
while true; do
   python client.py >> /home/candokevin/log.txt
done
