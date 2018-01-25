# Install pip
	sudo easy_install pip
# Install virtualenv 
	python -m pip install --user virtualenv
# Create virtual environment 
	python -m virtualenv env

# Activate virtual env
	source env/bin/activate
#  Deactivate
	deactivate

# Installing imports
	pip install requests 
	pip install package_name
# Install all imports at once 
	Add all your imports in a single txt file, say requirements.txt and every time you run 
	your program on a new system, just do a

 * pip install -r requirements.txt
	
