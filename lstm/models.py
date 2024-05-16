import torch
import torch.nn as nn
import torchvision.models as models


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        batch_size, seq_length, _, _, _ = x.size()
        x = x.view(-1, 3, 224, 224)  # Reshape for ResNet input
        x = self.resnet(x)
        x = x.view(batch_size, seq_length, -1)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, h0)
        out = self.dropout(out)
        
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# class RNNModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob = .5):
#         super(RNNModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout_prob = dropout_prob
        
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(dropout_prob)

#         self.fc1 = nn.Linear(hidden_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, num_classes)
    
#     def forward(self, x):
#         # Initialize hidden state with zeros
#         batch_size, seq_length, height, width, channels = x.size()
#         x = x.view(batch_size, seq_length, height * width * channels)
        
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Forward propagate RNN
#         out, _ = self.rnn(x, h0)
#         out = self.dropout(out)
        
#         # Decode the hidden state of the last time step
#         out = self.fc1(out[:, -1, :])
#         out = self.relu(out)
#         out = self.dropout(out)

#         out = self.fc2(out)
#         return out
    
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout_prob = dropout_prob
        
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.fc1 = nn.Linear(hidden_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         print(x.size)
#         # Reshape the input tensor to (batch_size, seq_length, input_size)
#         batch_size, seq_length, height, width, channels = x.size()
#         x = x.view(batch_size, seq_length, height * width * channels)
        
#         # Initialize hidden state and cell state with zeros
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
#         # Forward propagate LSTM
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.dropout(out)
        
#         # Decode the hidden state of the last time step
#         out = self.fc1(out[:, -1, :])
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
        
#         print(f"Output size: {out.size()}")

#         return out
    
class LSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        batch_size, seq_length, _, _, _ = x.size()
        x = x.view(-1, 3, 224, 224)  # Reshape for ResNet input
        x = self.resnet(x)
        x = x.view(batch_size, seq_length, -1)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out