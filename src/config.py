import yaml

class Config:
    def __init__(self, config_path="cfg.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Assign values
        self.gee = config["gee"]
        self.paths = config["paths"]
        self.training = config["training"]
        self.model = config["model"]

# Instantiate configuration
config = Config()
