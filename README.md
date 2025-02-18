## Infra structure

2 main use cases supported:

1 - messing around in a notebook to iterate on new ideas.
2 - running experiments from config files

dir structure:

experiments/ - responsible for taking a config file and running an experiment based on it, consisting of a single run or many runs (ie a hyperparam sweep). also manages files for the experiment. the notebook-based use case will not need any of this

components/ - the various things that are needed to run an experiment, eg models, dataloaders, tokenizers

tasks/ - things that you might do during an experiment, eg train a model, run an eval
