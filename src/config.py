

import yaml

class Config:
    def __init__(self, config_path="./src/cfg.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.gee = config["gee"]
        self.paths = config["paths"]
        self.training = config["training"]
        self.loss = config["losses"]
        self.metrics = config["metrics"]

config = Config()
