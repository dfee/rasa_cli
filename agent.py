#! /usr/bin/env python3

import io
import json
import logging
import pathlib
import warnings

import click
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu.training_data import load_data


HERE_DIR = pathlib.Path(__file__).parent.absolute()
NLU_MODELS_DIR = HERE_DIR.joinpath('models/nlu')
DIALOGUE_MODELS_DIR = HERE_DIR.joinpath('models/dialogue')


def _make_agent(model_dir=None, project_name='default'):
    """
    Given a model_dir and project_name, return an agent.
    :param model_dir: the model directory (e.g. "./models/")
    :param project_name: the policy project name
    """
    if not model_dir:
        project_path = NLU_MODELS_DIR.joinpath(project_name)
        model_dir = sorted(project_path.iterdir())[-1]
    return Agent.load(
        DIALOGUE_MODELS_DIR,
        interpreter=str(model_dir),  # strangely, doesn't support pathlib
    )


@click.group()
def cli():
    logging.basicConfig(level="INFO")
    warnings.filterwarnings('ignore')


@cli.command()
@click.option(
    '--resource-file', '-r',
    required=True, type=click.Path(exists=True),
)
@click.option(
    '--project-name', '-p',
    type=click.STRING, default='default',
)
def train_nlu(resource_file, project_name):
    training_data = load_data(resource_file)
    pipeline = [
        {"name": "nlp_spacy"},
        {"name": "tokenizer_spacy"},
        {"name": "intent_featurizer_spacy"},
        {"name": "intent_classifier_sklearn"},
    ]
    trainer = Trainer(RasaNLUModelConfig({"pipeline": pipeline}))
    interpreter = trainer.train(training_data)
    model_directory = trainer.persist(
        path=NLU_MODELS_DIR,
        project_name=project_name,
    )
    click.echo(
        "NLU model saved to '{}'".format(
            pathlib.Path(model_directory).\
            relative_to(HERE_DIR)
        )
    )


@cli.command()
@click.option(
    '--domain-file', '-d',
    required=True, type=click.Path(exists=True),
)
@click.option(
    '--stories-file', '-s',
    required=True, type=click.Path(exists=True),
)
def train_policy(domain_file, stories_file):
    agent = Agent(domain_file, policies=[KerasPolicy()])
    training_data = agent.load_data(stories_file)
    agent.train(
            training_data,
            validation_split=0.0,
            epochs=400
    )
    agent.persist(DIALOGUE_MODELS_DIR)
    click.echo(
        "Policy model saved to '{}'".format(
            pathlib.Path(DIALOGUE_MODELS_DIR).\
            relative_to(HERE_DIR)
        )
    )


@cli.command()
@click.option(
    '--model-dir', '-m',
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    '--project-name', '-p',
    type=click.STRING, default='default'
)
def chat(model_dir, project_name):
    agent = _make_agent(model_dir, project_name)
    print("Your bot is ready to talk! Type your messages here or send 'stop'")
    while True:
        in_ = input()
        if in_ == 'stop':
            break
        responses = agent.handle_message(in_)
        for response in responses:
            print(response['text'])


@cli.command()
@click.option(
    '--model-dir', '-m',
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    '--project-name', '-p',
    type=click.STRING, default='default'
)
@click.option(
    '--stories-file', '-s',
    required=True, type=click.Path(exists=True),
)
@click.option(
    '--max-history', '-h',
    type=click.INT, default=2,
)
@click.argument('outfile', type=click.Path(writable=True))
def visualize(model_dir, project_name, stories_file, max_history, outfile):
    agent = _make_agent(model_dir, project_name)
    agent.visualize(
        resource_name=stories_file,
        output_file=outfile,
        max_history=max_history,
    )


if __name__ == '__main__':
    cli()
