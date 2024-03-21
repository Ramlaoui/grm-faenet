import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from src.train import Trainer

@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def main(config: DictConfig):
    print("Configuration used:")
    print(OmegaConf.to_yaml(config))

    debug = config.get("debug", False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer = Trainer(config_dict, debug=debug, device=device)
    trainer.train()

if __name__ == "__main__":
    main()