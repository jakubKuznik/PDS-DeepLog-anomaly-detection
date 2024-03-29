
# TODO find out if i really need executable

target: log-monitor.py
	log-monitor

install: requirements.txt
	pip3 install -r requirements.txt

# TODO 
zip:
	xlkuzni04.zip

clean:
	rm -rf __pycache__
