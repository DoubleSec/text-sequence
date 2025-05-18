"""All that happens here is that we run the "initial_prep" function using values from the config."""

import yaml

from src.data import initial_prep

if __name__ == "__main__":

    with open("./cfg/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    initial_prep(**config["preparation_params"])
