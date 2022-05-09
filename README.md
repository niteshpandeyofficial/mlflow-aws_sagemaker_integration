Create requirements file
```bash
touch requirements.txt
or 
pip freeze > requirements.txt
```

install the library
```bash 
 pip install -r requirements.txt
```

to run the programfile
```bash
python main.py
```

create empty artifacts folder
```bash
touch artifacts
```

to check the details using mlflow user interface
```bash
mlfow ui
```

required application to start using this project

mysql workbench--to coonect with remote dbs
having aws account

steps:

create s3 bucket
create Relational dbs using aws dbs services

create policy (action allowed in s3-read,write,list,delete)
create role and assign created policy to role



