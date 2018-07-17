## rasa_cli

This is a template integrating the [`click`](https://github.com/pallets/click) command line suite and the [`rasa`](https://github.com/RasaHQ) chat agent.


### Quickstart

1. Train the **interpreter** (rasa_nlu)

    ```
    ./agent.py train_nlu -r data/nlu.md
    ```

2. Train the **policy** (keras)

    ```
    ./agent.py train_policy -d data/domain.yaml -s data/stories.md
    ```

3. Chat with the agent

    ```
    ./agent.py chat
    ```

### Visualization

After training the **interpreter** and training the **policy**, you can visualize the graph by running (note, this requires installing graphviz as descriped in `Setup` below):

```
./agent.py visualize -s data/stories.md story-graph.png
```

### Setup

These setup instructions assume a modern version of Python 3.5+, but it should work on Python 2.7+ or Python 3.3+.

1. clone the repository, and enter the directory:

    ```
    git clone https://github.com/dfee/rasa_nlu_cli.git && cd rasa_nlu_cli
    ```

2. create a virtualenv (and we'll activate it too):

    ```
    python3 -m venv env && source env/bin/activate
    ```

3. install the dependencies:

    ```
    pip install -r requirements.txt
    ```

4. install the spacy language files:

    ```
    python -m spacy download en
    ```

5. (optional) install `graphviz` to enable visualization

    For MacOS (assuming [homebrew](http://brew.sh) is installed):
    ```
    brew install graphviz && pip install pygraphviz
    ```

    For Ubuntu (or anything in the Debian family using `apt`):
    ```
    apt-get -qq install -y graphviz libgraphviz-dev pkg-config && pip install pygraphviz
    ```
