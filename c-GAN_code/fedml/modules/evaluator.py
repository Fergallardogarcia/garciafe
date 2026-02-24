"""Evaluation function to test the model performance."""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

def evaluate(
        model,
        testloader: torch.utils.data.DataLoader,
        device: str,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[float, float]:
    
    """Validate the model on the entire test set.
    
    :param model: The local model that needs to be evaluated.
    :param testloader: The dataloader of the dataset to use for evaluation.
    :param device: The device to evaluate the model on i.e. cpu or cuda. 
    :param criterion: The loss function to use for model evaluation.
    :returns: Evaluation loss and accuracy of the model.
    """
    if criterion is None: criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss += criterion(outputs, target).item() * target.size(0)
            _, predicted = torch.max(outputs.data, 1)  
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    loss /= total
    return loss, accuracy, total


def evaluate_gan(
        dis_model: nn.Module,
        testloader: torch.utils.data.DataLoader,
        device: str,
        num_classes,
        num_sample_per_class,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[float, float, List[float]]:
    """Validate the model on the entire test set.
    
    :param dis_model: The local model that needs to be evaluated.
    :param testloader: The dataloader of the dataset to use for evaluation.
    :param device: The device to evaluate the model on i.e. cpu or cuda. 
    :param criterion: The loss function to use for model evaluation.
    :returns: Evaluation loss and accuracy of the model.
    """
    if criterion is None: criterion = nn.CrossEntropyLoss()

    # Stage the discriminator model to the run device
    if next(dis_model.parameters()).device != device:
        dis_model.to(device)
    dis_model.eval()

    # Variables to hold evaluation stats
    total_loss = 0.0
    accu_cls = [0 for _ in range(num_classes)]
    accu_all = 0.0

    dis_model.eval()
    with torch.no_grad(): 
        # Generate samples with given z and l values
        # and use them to evaluate current model
        for samples, labels in testloader:
            samples, labels = samples.to(device), labels.to(device)
            dis_predict = dis_model(samples)
            _, preds = torch.max(dis_predict.data, dim=1)

            # Compute Loss
            g_loss = criterion(dis_predict, labels)
            total_loss += g_loss.item() * len(labels)
            
            # Compute accuracy per class
            for c in range(num_classes):
                accu_cls[c] += ((preds == labels) * (labels == c)).float().sum().item()
            
            # Compute overall accuracy
            accu_all += (preds == labels).float().sum().item()

    # Updates statistics
    avg_loss = total_loss / (num_sample_per_class * num_classes)
    accu_all = accu_all / (num_sample_per_class * num_classes)
    accu_cls = [c / num_sample_per_class for c in accu_cls]

    return avg_loss, accu_all, accu_cls