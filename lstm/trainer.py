import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import wandb
from dataset import create_data_loaders


class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train(self, dataset, train_indices, test_indices,  \
        clip_gradient=False, clip_value=5, save_interval=5, eval_interval=1, log_file="training_log.csv",\
        regularization=True, name = None, hp_tune=False, exp_label=None, phase_exps=False, phase_exps_train = None, phase_exps_test = None
        ):
        if hp_tune:
            wandb.init(project="cs518_lstm", 
                       name="initial_sweep"
            )
            learning_rate = wandb.config.learning_rate
            batch_size = wandb.config.batch_size
            l2_lambda = wandb.config.l2_lambda
            num_epochs = 7
            train_loader = create_data_loaders(dataset, batch_size=batch_size, num_workers=0, indices=train_indices, train=True)
            test_loader = create_data_loaders(dataset, batch_size=2, num_workers=0, indices=test_indices)
        else:
            learning_rate = .0001
            batch_size = 8
            l2_lambda = .0001
            num_epochs = 12
            if phase_exps:
                train_loader = create_data_loaders(phase_exps_train, batch_size=batch_size, num_workers=0, indices=train_indices, train=True,phase_exps=phase_exps)
                test_loader = create_data_loaders(phase_exps_test, batch_size=2, num_workers=0, indices=test_indices, phase_exps=phase_exps)
                print(len(train_loader))
                print(len(test_loader))
            
            else:

                train_loader = create_data_loaders(dataset, batch_size=batch_size, num_workers=0, indices=train_indices, train=True,phase_exps=phase_exps)
                test_loader = create_data_loaders(dataset, batch_size=2, num_workers=0, indices=test_indices, phase_exps=phase_exps)

        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        # Initialize logging dataframe
        log_data = []

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
            for frames, skill_levels in progress_bar:
                frames = frames.float()
                frames = frames.to(self.device)
                skill_levels = skill_levels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(frames)
                loss = criterion(outputs, skill_levels)
                # print(loss)
                if regularization:
                    l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                    loss = loss + l2_lambda * l2_reg
                    
                # print('about to backpropagate')
                loss.backward()

                if clip_gradient:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                optimizer.step()
                running_loss += loss.item()

                progress_bar.update(1)

            epoch_loss = running_loss / len(train_loader)

            if (epoch + 1) % save_interval == 0:
                torch.save(self.model.state_dict(), f"models_save/model_epoch_{epoch+1}_{name}_{exp_label}.pth")

            print("EVALUATING")
            if (epoch + 1) % eval_interval == 0:
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for frames, skill_levels in test_loader:
                        frames = frames.float()
                        frames = frames.to(self.device)
                        skill_levels = skill_levels.to(self.device)
                        outputs = self.model(frames)
                        _, predicted = torch.max(outputs, 1)
                        total += skill_levels.size(0)
                        correct += (predicted == skill_levels).sum().item()

                accuracy = correct / total
                print(accuracy)
                if hp_tune:
                    # Log metrics to Wandb
                    wandb.log({"Loss": epoch_loss, "Accuracy": accuracy})

            log_data.append({"Epoch": epoch+1, "Loss": epoch_loss, "Accuracy": accuracy})

        save_name = name + '_' + exp_label + '.csv'
        pd.DataFrame(log_data).to_csv(save_name, index=False)
        print("Training finished!")

        return self.model



