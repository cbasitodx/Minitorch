SHELL := /bin/bash

all:
	@echo "\n USE 'make env' TO GENERATE THE VIRTUALENV FOR THIS PROJECT! 👨‍💻 💬 THIS IS THE FIRST COMMAND YOU SHOULD RUN! 💬 \n"
	@echo "\n USE 'make doc' TO GENERATE DOCUMENTATION FOR THIS PROJECT! DOCUMENTATION IS AVAILABLE IN docs/__build/html 📁  \n"
	@echo "\n USE 'make tests' TO RUN TESTS FOR THIS PROJECT! 🧪"

# Creates documentation
doc:
	(\
		@echo "\n --------------- ACTIVATING VIRTUALENV --------------- \n"; \
		source venv/bin/activate; \
		@echo "\n --------------- CREATING DOCUMENTATION --------------- \n"; \
		sphinx-apidoc -o ./docs ./minitorch; \
		make -C ./docs html; \
		@echo "\n --------------- DOCUMENTATION CREATED! (check log) --------------- \n"; \
		deactivate; \
	)

# Creates virtualen and installs dependencies
env:
	(\ 
		@echo "\n --------------- CREATING VIRTUALENV --------------- \n"; \
		virtualenv venv; \
		@echo "\n --------------- VIRTUALENV CREATED --------------- \n"; \
		@echo "\n --------------- ACTIVATING VIRTUALENV --------------- \n"; \
		source venv/bin/activate; \
		@echo "\n --------------- VIRTUALENV ACTIVATED --------------- \n"; \
		@echo "\n --------------- INSTALLING DEPENDENCIES --------------- \n"; \
		pip3 install -r requirements.txt; \
		@echo "\n --------------- DEPENDENCIES INSTALLED --------------- \n"; \
		deactivate; \
	)

# TODO: PONER UN all (con emojis bonitos q explique el funcionamiento), PONER TAMBIEN UN tests (para hacer tests unitarios!), Y ṔONER UN CLEAN!!