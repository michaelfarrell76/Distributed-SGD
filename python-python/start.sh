#! /bin/bash
#! /home/candokevin/anaconda2/bin/python
cd /home/candokevin/stash/distributed-sgd/python-python
git pull
rm /home/candokevin/log.txt
while true; do
   python client.py >> /home/candokevin/log.txt
done
