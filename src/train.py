import torch
import wandb
import tensorboardX
from tqdm import tqdm
from copy import deepcopy
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from .datasets.base_dataset import BaseDataset, ParallelCollater
from .modules.frame_averaging import FrameAveraging
from .faenet import FAENet
from .datasets.data_utils import Normalizer
from torchvision.transforms import Compose

def transformations_list(config):
    transform_list = []
    if config.get('equivariance', "") != "":
        transform_list.append(FrameAveraging(config['equivariance'], config['fa_type'], config['dataset'].get('oc20', True)))
    if len(transform_list) > 0:
        return Compose(transform_list)
    else:
        return None


class Trainer():
    def __init__(self, config, device="cpu", debug=False):
        self.config = config
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run_name = f"{self.config['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.device
        self.load()

    def load(self):
        self.load_model()
        self.load_logger()
        self.load_optimizer()
        self.load_train_loader()
        self.load_val_loaders()
        self.load_scheduler()
        self.load_criterion()
    
    def load_logger(self):
        if not self.debug:
            if self.config['logger'] == 'wandb':
                wandb.init(project=self.config['project'], name=self.run_name)
                wandb.config.update(self.config)
                self.writer = wandb
            elif self.config['logger'] == 'tensorboard':
                self.writer = tensorboardX.SummaryWriter(log_dir=f"runs/{self.run_name}")
    
    def load_model(self):
        self.model = FAENet(**self.config["model"]).to(self.device)
    
    def load_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer'].get('lr_initial', 1e-4))
    
    def load_scheduler(self):
        if self.config['optimizer'].get('scheduler', None) == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config['optimizer'].get('epochs', 1)*len(self.train_loader))
        else:
            self.scheduler = None
    
    def load_criterion(self, reduction='mean'):
        if self.config['model'].get('loss', 'mse') == 'mse':
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        elif self.config['model'].get('loss', 'mse') == 'mae':
            self.criterion = torch.nn.L1Loss(reduction=reduction)
    
    def load_train_loader(self):
        if self.config['dataset']['train'].get("normalize_labels", False):
            if self.config['dataset']['train']['normalize_labels']:
                self.normalizer = Normalizer( mean=self.config['dataset']['train']['target_mean'], std=self.config['dataset']['train']['target_std'])
            else:
                self.normalizer = None

        self.parallel_collater = ParallelCollater() # To create graph batches
        transform = transformations_list(self.config)
        train_dataset = BaseDataset(self.config['dataset']['train'], transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["optimizer"]['batch_size'], shuffle=True, num_workers=0, collate_fn=self.parallel_collater)
    
    def load_val_loaders(self):
        self.val_loaders = []
        for split in self.config['dataset']['val']:
            transform = transformations_list(self.config)
            val_dataset = BaseDataset(self.config['dataset']['val'][split], transform=transform)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config["optimizer"]['eval_batch_size'], shuffle=False, num_workers=0, collate_fn=self.parallel_collater)
            self.val_loaders.append(val_loader)
    
    def faenet_call(self, batch):
        equivariance = self.config.get("equivariance", "")
        outputs = []
        if equivariance == "data_augmentation":
            return self.model(batch)
        else:
            original_batch = deepcopy(batch)
            # The frame's positions are computed by the data loader
            # If stochastic frame averaging is used, fa_pos will only have one element
            # If the full frame is used, it will have 8 elements
            for i in range(len(batch.fa_pos)):
                batch = deepcopy(original_batch)
                batch.pos = batch.fa_pos[i]
                output = self.model(batch)
                outputs.append(output["energy"])
        energy_prediction = torch.stack(outputs, dim=0).mean(dim=0)
        output["energy"] = energy_prediction
        return output

    def train(self):
        log_interval = self.config["optimizer"].get("log_interval", 100)
        epochs = self.config["optimizer"].get("epochs", 1)
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(self.train_loader)
            for batch_idx, (batch) in enumerate(pbar):
                batch = batch[0].to(self.device)
                self.optimizer.zero_grad()
                output = self.faenet_call(batch)
                target = batch.y_relaxed
                if self.normalizer:
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target
                loss = self.criterion(output["energy"].reshape(-1), target_normed.reshape(-1))
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                metrics = {
                    "train/loss": loss.detach().item(),
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch,
                }
                if not self.debug:
                    self.writer.log(metrics)
                pbar.set_description(f'Epoch {epoch+1}/{epochs} - Loss: {loss.detach().item():.6f}')
                if self.scheduler:
                    self.scheduler.step()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if epoch != epochs-1:
                self.validate(epoch, splits=[0]) # Validate on the first split (val_id)
        self.validate(epoch) # Validate on all splits
        
        
    def validate(self, epoch, splits=None):
        self.model.eval()
        for i, val_loader in enumerate(self.val_loaders):
            if splits and i not in splits:
                continue
            split = list(self.config['dataset']['val'].keys())[i]
            pbar = tqdm(val_loader)
            total_loss = 0
            for batch_idx, (batch) in enumerate(pbar):
                batch = batch[0].to(self.device)
                output = self.faenet_call(batch)
                target = batch.y_relaxed
                if self.normalizer:
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target
                loss = self.criterion(output["energy"].reshape(-1), target_normed.reshape(-1))
                total_loss += loss.detach().item()
                pbar.set_description(f'Val {i} - Epoch {epoch+1} - Loss: {loss.detach().item():.6f}')
            total_loss /= len(val_loader)
            if not self.debug:
                self.writer.log({f"{split}/loss": total_loss, f"{split}/epoch": epoch})