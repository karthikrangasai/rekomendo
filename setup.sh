echo -e "\e[1m>>> Running checks for rekomendo"
echo -e "\e[0m"

# Check for virtualenv
echo -ne "\e[1m>> Checking if 'virtualenv' is present: "
venv_bool=$(pip3 freeze | grep virtualenv)

if [[ $venv_bool == "" ]]
then
	echo -e "\e[0mNo.\nStarting Download"
	pip3 install virtualenv
else
	echo -e "\e[0mYes."
fi

mkdir dataset
mkdir dataset/pickle_files

# Check for dataset
echo -ne "\e[1m>> Checking if dataset is present: "
DATASET=./ml-1m.zip
if [[ -f "$DATASET" ]]
then
	echo -e "\e[0mYes."
	unzip -u ml-1m.zip
else
	echo -e "\e[0mNo.\nPlease check the dataset file."
fi

# Check for virtual environment
echo -ne "\e[1m>> Checking if virtual environment is present: "
if [[ -d "./env" ]]
then
        echo -e "\e[0mYes."
else
        echo -e "\e[0mNo.\nCreating Virtual Environemt"
        virtualenv env
fi

source "env/bin/activate"
pip install -r requirements.txt
python setup.py
deactivate