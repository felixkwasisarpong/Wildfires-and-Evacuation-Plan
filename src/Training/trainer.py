import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from sklearn.model_selection import train_test_split
from models.asufm import ASUFM
from models.asu import ASU
from models.stgnn import STGNN
from models.Unet import UNet
from utils.losses import DiceLoss, FocalLoss, TverskyLoss, IoULoss, BCELoss
from utils.metrics import precision, recall, f1, pr_auc, dice_coefficient, intersection_over_union
from dataloader.dataloadermodule import WildfireDataset
from dataloader.preprocessing import Preprocessor
import shap
import matplotlib.pyplot as plt
from config import config
import ee

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
class Trainer:
    def __init__(self, config):
        self.model = self.get_model(config.training["model_name"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training["learning_rate"], weight_decay=config.training["weight_decay"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.training["num_epochs"], eta_min=0)
        
        # Define loss functions
        if config.training["model_name"] == "STGNN":
            self.loss_stage1 = BCELoss()
            self.loss_stage2 = lambda preds, targets: TverskyLoss()(preds, targets) + IoULoss()(preds, targets)
        else:
            self.loss_stage1 = BCELoss()
            self.loss_stage2 = FocalLoss()

        # Initialize the FireSpreadDataModule
        self.data_module = WildfireDataset(
            data_dir=config.paths["hd5f_path"],
            batch_size=config.training["batch_size"],
            n_leading_observations=config.training["n_leading_observations"],
            n_leading_observations_test_adjustment=config.training["n_leading_observations_test_adjustment"],
            crop_side_length=config.training["crop_side_length"],
            load_from_hdf5=config.training["load_from_hdf5"],
            num_workers=config.training["num_workers"],
            remove_duplicate_features=config.training["remove_duplicate_features"],
            features_to_keep=config.training["features_to_keep"],
            return_doy=config.training["return_doy"],
            data_fold_id=config.training["data_fold_id"]
        )
        self.data_module.setup()

        # Get dataloaders
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()

        self.loss_history = []
        self.val_loss_history = []
        self.pr_auc_history = []
        self.val_pr_auc_history = []

        # Call explain_model after full_dataset is defined
        self.explain_model()

    def get_model(self, model_name):
        if (model_name == "ASUFM"):
            return ASUFM()
        elif (model_name == "ASU"):
            return ASU()
        elif (model_name == "STGNN"):
            return STGNN(in_channels=config.training["inn_channels"], out_channels=config.training["out_channels"], num_nodes=config.training["num_nodes"], num_timesteps=config.training["num_timesteps"])
        else:
            raise ValueError("Invalid model name!")

    def train(self):
        for epoch in range(config.training["num_epochs"]):
            self.model.train()
            criterion = self.loss_stage1 if epoch < config.training["pretrain_epochs"] else self.loss_stage2
            train_loss = 0

            for images, targets in self.train_loader:
                preds = self.model(images)
                loss = criterion(preds, targets)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_loss += loss.item()

            self.scheduler.step()
            avg_train_loss = train_loss / len(self.train_loader)
            self.loss_history.append(avg_train_loss)

            print(f"Epoch {epoch}, Training Loss: {avg_train_loss}")

            # Run validation
            val_metrics, avg_val_loss = self.validate()
            self.val_loss_history.append(avg_val_loss)
            self.val_pr_auc_history.append(val_metrics["PR-AUC"])

            print(f"Validation Metrics: {val_metrics}")

        self.save_plots()
        self.test()  # Run final test

    def validate(self):
        """ Run validation on the validation dataset """
        self.model.eval()
        criterion = self.loss_stage2  # Use stage 2 loss for validation
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                preds = self.model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                
                all_preds.append(preds)
                all_targets.append(targets)

        avg_val_loss = val_loss / len(self.val_loader)

        # Compute metrics
        val_metrics = {
            "Precision": precision(all_preds, all_targets),
            "Recall": recall(all_preds, all_targets),
            "F1 Score": f1(all_preds, all_targets),
            "PR-AUC": pr_auc(all_preds, all_targets),
            "Dice Coefficient": dice_coefficient(all_preds, all_targets),
            "IoU": intersection_over_union(all_preds, all_targets)
        }

        return val_metrics, avg_val_loss

    def test(self):
        """ Run final test after training """
        self.model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.test_loader:
                preds = self.model(images)
                loss = self.loss_stage2(preds, targets)
                test_loss += loss.item()
                
                all_preds.append(preds)
                all_targets.append(targets)

        avg_test_loss = test_loss / len(self.test_loader)

        # Compute metrics
        test_metrics = {
            "Precision": precision(all_preds, all_targets),
            "Recall": recall(all_preds, all_targets),
            "F1 Score": f1(all_preds, all_targets),
            "PR-AUC": pr_auc(all_preds, all_targets),
            "Dice Coefficient": dice_coefficient(all_preds, all_targets),
            "IoU": intersection_over_union(all_preds, all_targets)
        }

        print(f"Final Test Loss: {avg_test_loss}")
        print(f"Final Test Metrics: {test_metrics}")

    def save_plots(self):
        """ Save training and validation plots """
        plt.figure()
        plt.plot(range(config.training["num_epochs"]), self.loss_history, label="Training Loss")
        plt.plot(range(config.training["num_epochs"]), self.val_loss_history, label="Validation Loss", linestyle="dashed")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curve")
        plt.legend()
        plt.savefig("loss_plot.png")

        plt.figure()
        plt.plot(range(config.training["num_epochs"]), self.pr_auc_history, label="Training PR-AUC")
        plt.plot(range(config.training["num_epochs"]), self.val_pr_auc_history, label="Validation PR-AUC", linestyle="dashed")
        plt.xlabel("Epoch")
        plt.ylabel("PR-AUC")
        plt.title("PR-AUC Curve")
        plt.legend()
        plt.savefig("pr_auc_plot.png")

    def explain_model(self):
        explainer = shap.GradientExplainer(self.model, self.full_dataset[:100])
        shap_values = explainer.shap_values(self.full_dataset[:100])
        shap.summary_plot(shap_values, self.full_dataset[:100], feature_names=["Band " + str(i) for i in range(self.full_dataset.num_features)])

