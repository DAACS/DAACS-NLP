# DAACS-NLP

## project set up

### 1 download

git clone wherever you want to run this project from

### 2 build your python virtualenv 

```
python3 -m venv .venv     # build your project virtual env
source .venv/bin/activate # activate your virual env
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r ./requirements.txt
```

### Clean you pip packages

Do this every now and then!  This uninstall/reinstalls all pip packages back to exactly what is in requirements.txt.

```
pip freeze | xargs pip uninstall -y

```