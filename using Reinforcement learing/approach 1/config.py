  
from typing import Any
import re
import yaml
import json


def load_yaml(path):
    # Fix yaml numbers https://stackoverflow.com/a/30462009/11037553
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)

class TrainingConfig():
    def __init__(self, config:dict=None)-> None:
        self.batch_size = config.get("batch_size")
        self.learning_rate = config.get("learning_rate")
        self.loss = config.get("loss")
        self.num_episodes = config.get("num_episodes")
        self.train_steps = config.get("train_steps")
        self.warmup_episode = config.get("warmup_episode")
        self.save_freq = config.get("save_freq")

class OptimizerConfig():
    def __init__(self, config:dict=None)-> None:
        self.name = config.get("name")
        self.lr_min = config.get("lr_min")
        self.lr_decay = config.get("lr_decay")

class RlConfig():
    def __init__(self, config:dict=None) -> None:
        self.gamma = config.get("gamma")
        self.max_steps_per_episode = config.get("max_steps_per_episode")
        self.target_model_update_episodes = config.get("target_model_update_episodes")
        self.max_queue_length = config.get("max_queue_length")

class EpsilonConfig():
    def __init__(self, config:dict=None) -> None:
        self.max_epsilon = config.get("max_epsilon")
        self.min_epsilon = config.get("min_epsilon")
        self.decay_epsilon = config.get("decay_epsilon")

class Config:
    """ User config class """
    def __init__(self, path: str=None):        
        if path is not None:
            config = load_yaml(path)
            self.training = TrainingConfig(config.get("training", {}))
            self.optimizer = OptimizerConfig(config.get("optimizer", {}))
            self.rl = RlConfig(config.get("rl", {}))
            self.epsilon = EpsilonConfig(config.get("epsilon", {}))