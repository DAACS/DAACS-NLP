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

Now you can activate and deactivate that specific python env and path like this: 
```
user@host: source ./.venv/bin/activagte
user@host: deactivate 

```

### Clean you pip packages

Do this every now and then!  This uninstall/reinstalls all pip packages back to exactly what is in requirements.txt.

```
pip freeze | xargs pip uninstall -y

```

### Edit your .env file.
Make sure you have python path and OPENAPI_KEY in there, like this. 
```sh
cat ./.env

PYTHONPATH=/Users/afraser/Documents/src/daacs-nlp/src:/Users/afraser/Documents/src/daacs-nlp/.venv/lib/python3.11/site-packages
OPENAI_API_KEY=sk-2r7jkiZZNSdJEBrZKnkuT3Blb*******STUFFGETYOUROWN*********
```

### Edit your .venv/bin/activate 
Should look about like this:

```sh
# ... stuff 
VIRTUAL_ENV="/Users/afraser/Documents/src/daacs-nlp/.venv"
export VIRTUAL_ENV

_OLD_VIRTUAL_PATH="$PATH"
PATH="$VIRTUAL_ENV/bin:$PATH"
export PATH
# ... enter it here, where other envs are set.
export PYTHONPATH=/Users/afraser/Documents/src/daacs-nlp/src:/Users/afraser/Documents/src/daacs-nlp/.venv/lib/python3.11/site-packages
##

```

export JAVA_HOME="/Users/afraser/.jdk/jdk-11.0.18+10/Contents/Home"
export SPARK_HOME="/Users/afraser/Documents/src/daacs-nlp/.venv/lib/python3.11/site-packages/pyspark"
export PYTHONPATH=/Users/afraser/Documents/src/daacs-nlp/.venv/bin/python:/Users/afraser/Documents/src/daacs-nlp/src:/Users/afraser/Documents/src/daacs-nlp/.venv/lib/python3.11/site-packages
