import argparse

def get_parameters():
    parser = argparse.ArgumentParser(description='Surgical Skill Prediction')
    
    parser.add_argument('--model', type=str, default='RNN', help='Model choice: RNN or LSTM')
 
    parser.add_argument('--mode', type=str, default='TRAIN', help='Mode: TRAIN or TEST')
    
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test set')
    
    parser.add_argument('--device_num', type=int, default=0, help='GPU ID')

    parser.add_argument('--label', type=str, help='which dataset to run')
    
    parser.add_argument('--phases', action="store_true", help='train/ test on easy hard phases')
    
    parser.add_argument('--train_easy', action="store_true", help='if phases, toggle this to train easy vs hard')
    
    args = parser.parse_args()
    return args