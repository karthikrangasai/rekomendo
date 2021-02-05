# rekomendo: Movie Recommender System
Movie recommendation system using the Movielens dataset.

## Setup, Installation and Usage
- METHOD 1:
	- Make sure you have `pip` and `virtualenv` installed on your systems (Examples given for ubuntu)
		- `sudo apt install python3-pip` : To install pip
		- `pip3 install virtualenv` : To install virtualenv
	- Execute the following commands from the root of the project to setup the project:
		- `virtualenv env` : Create python virutal environment for the project
		- `source env/bin/activate` : Activate the virtual environment for the project
		- `pip install numpy pandas` : Install the required dependencies for the project
		- `python setup.py` : Setup the project
		- `python app.py` : Run the project
		- `deactivate` : To deactivate the virtual environment (Do not deactivate the environemnt during any of the above steps)
- METHOD 2:
	- Make sure you have `pip` and `virtualenv` installed on your systems (Examples given for ubuntu)
		- `sudo apt install python3-pip` : To install pip
	- Execute the following commands from the root of the project to setup the project:
		- `bash setup.sh`: Setup the project
		- `bash run.sh`: Run the project

## Folder Strucuture
- Before setup:
```
.
├── app.py
├── ml-1m.zip
├── ratings_matrix_original.npy
├── README.md
├── requirements.txt
├── run.sh
├── setup.py
├── setup.sh
└── src
    ├── collaborative_model.py
    ├── cur_model.py
    ├── __init__.py
    └── svd_model.py
```