import yaml
import os
import datetime
# from src.data.datamodule import create_dataloaders
# from src.models.base_model import BaseModel
from src.training.trainer import Trainer

def run_training(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("experiments", timestamp)
    
    # train_loader, val_loader = create_dataloaders(cfg["data"])

    # model = BaseModel(**cfg["model"])
    model = model.to(cfg["training"]["device"])

    trainer = Trainer(
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        save_dir=save_dir
    )

    print(f"Logging to: {save_dir}")
    trainer.fit(model, train_loader, val_loader)
