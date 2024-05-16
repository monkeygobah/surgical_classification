import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from dataset import create_data_loaders
from loader import  SurgeryDataset
from trainer import Trainer
from tester import test
from models import RNNModel, LSTMModel
from parameters import get_parameters
from torch.nn.parallel import DistributedDataParallel, DataParallel
import wandb

if __name__ == '__main__':
    # Get command-line arguments
    args = get_parameters()
    hp_tune = False
    
    MODEL_CHOICE = args.model
    MODE = args.mode
    root_dir = args.root_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    test_size = args.test_size
    exp_label = args.label
    phase_exps = args.phases
    
    if phase_exps:
        train_easy = args.train_easy 
        easy_path = 'easy_phases/'
        hard_path = 'hard_phases/'
    
    # Define model, loss function, and optimizer
    if MODEL_CHOICE == "RNN":
        input_size = 224 * 224 * 3
        hidden_size = 512
        num_layers = 5
        num_classes = 2
        model = RNNModel(input_size, hidden_size, num_layers, num_classes)
    elif MODEL_CHOICE == "LSTM":
        input_size = 224 * 224 * 3
        hidden_size = 256
        num_layers = 2
        num_classes = 2
        dropout_prob = 0.5
        model = LSTMModel(hidden_size, num_layers, num_classes, dropout_prob)
        # model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_prob)

    else:
        raise ValueError(f"Invalid model choice: {MODEL_CHOICE}")
    
    
    # Set device
    device = torch.device(f"cuda:{str(args.device_num)}" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if phase_exps:
        if train_easy:
            train_dataset=SurgeryDataset(easy_path, interval=True)
            test_dataset=SurgeryDataset(hard_path, interval=True)
        else:
            test_dataset=SurgeryDataset(easy_path, interval=True)
            train_dataset=SurgeryDataset(hard_path, interval=True)  
        
        #dont need these but generate to avoid error later 
        train_indices, test_indices = train_test_split(list(range(len(test_dataset))), test_size=test_size, random_state=42)
        dataset=None
    
    else:
        # Load the dataset
        dataset = SurgeryDataset(root_dir, interval=True)
        # Split the dataset into training and testing sets
        train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=test_size, random_state=42)

       

    model.to(device)
    # model = DataParallel(model)


    if hp_tune:
        sweep_config = {
            'method': 'grid',
            'metric': {
                'name': 'Accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'values': [ 1e-4, 1e-3, 1e-2, 1e-1]
                },
                'batch_size': {
                    'values': [8,16,32]
                },
                'l2_lambda': {
                    'values': [ 1e-4, 1e-3, 1e-2]
                }

            }
        }
        
        # dont pass dataloader because will dynamically load data based on batch size hp 
        sweep_id = wandb.sweep(sweep=sweep_config, project="cs518_lstm")

        def train_wrapper():

            trainer = Trainer(model,device)
            trainer.train(dataset, train_indices, test_indices, name=MODEL_CHOICE, hp_tune=hp_tune, exp_label = exp_label)
        wandb.agent(sweep_id, function=train_wrapper, count=40)

        test(model, test_loader, device)

    else:
        trainer = Trainer(model,device)
        trainer.train(dataset, train_indices, test_indices, name=MODEL_CHOICE, hp_tune=hp_tune, exp_label = exp_label, \
            phase_exps=phase_exps, phase_exps_train = train_dataset, phase_exps_test = test_dataset)


    # if MODE == 'TRAIN':
    #     # Train the model
    #     model.to(device)

    #     model = DataParallel(model)

    #     model = train(model, train_loader,test_loader, criterion, optimizer, device, num_epochs, name=MODEL_CHOICE)
    
    # # elif MODE == 'TEST':
    #     # model_path = args.model_path
    #     # model.load_state_dict(torch.load(model_path))
    #     model.eval()
        
    #     # Test the model
    #     test(model, test_loader, device)
    # else:
    #     raise ValueError(f"Invalid mode: {MODE}")