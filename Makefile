venv: 
	python -m venv venv
requirements:
	pip install -r requirements.txt
clean:
	black src
	black scripts
	isort src
	isort scripts
data_prep:
	python scripts/data_prep.py --SAMPLE_SIZE 3000 
train:
	python scripts/train.py
clear_logs:
	rm -f logs/* 