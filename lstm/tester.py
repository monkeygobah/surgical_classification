import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def test(model, test_loader, device):
    model.eval()
    
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.float()
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")