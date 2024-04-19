

install:
	pip3 install -r requirements.txt

run: 
	python3 src/log-monitor.py -training logs/small-HDFS-annotate.log -testing logs/test-1-test.log
