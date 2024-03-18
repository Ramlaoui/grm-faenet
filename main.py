import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from src.train import Trainer

@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def main(config: DictConfig):
    print("Configuration used:")
    print(OmegaConf.to_yaml(config))

    is_debug = config.get("is_debug", False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer = Trainer(config_dict, is_debug=is_debug, device=device)
    trainer.train()

if __name__ == "__main__":
    main()