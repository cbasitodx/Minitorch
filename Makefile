doc:
	echo -e "\n --------------- CREANDO DOCUMENTACIÓN --------------- \n"
	sphinx-apidoc -o ./docs ./minitorch
	make -C ./docs html
	echo -e "\n --------------- DOCUMENTACIÓN CREADA --------------- \n"