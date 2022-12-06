import torch
import math
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from preprocessing import load_official_traintest_split
from util_functions import MyTrainDataset, MyTestDataset, precision_recall_calculation
from igmc_model import IGMC

default_params = {'Epochs':40, 'Batch_Size':50, 'LR': 1e-3, 'LR_Decay_Step' : 20, 'LR_Decay_Value' : 10, 'Max_Nodes_Per_Hop':200}

# function used to train the IGMC model on the dataset 
# returns training losses, model to test and the test dataset loader.

def train(fname = 'ml_100k', params = default_params):
    
    EPOCHS = params['Epochs']
    BATCH_SIZE = params['Batch_Size']
    LR_DECAY_STEP = params['LR_Decay_Step']
    LR_DECAY_VALUE = params['LR_Decay_Value']
    LR = params['LR']
    MAX_NODES_PER_HOP = params['Max_Nodes_Per_Hop']
    
    #setting the seed value to be able to replicate results
    torch.manual_seed(123)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.synchronize()
        device = torch.device('cuda')
        
    #get the train, test split of the dataset
    (adj_train, train_labels, train_u_indices, train_v_indices, test_labels, test_u_indices, test_v_indices, class_values
    ) = load_official_traintest_split(fname, testing=True)

    
    train_dataset = eval('MyTrainDataset')(root='data/ml_100k/testmode/train', A=adj_train, 
        links=(train_u_indices, train_v_indices), labels=train_labels, h=1, max_nodes_per_hop=MAX_NODES_PER_HOP, class_values=class_values)
    test_dataset = eval('MyTestDataset')(root='data/ml_100k/testmode/test', A=adj_train, 
        links=(test_u_indices, test_v_indices), labels=test_labels, h=1, max_nodes_per_hop=MAX_NODES_PER_HOP, class_values=class_values)
        
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = IGMC()
    model.to(device)
    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=0)
    
    train_losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss_all = 0
        for index, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            train_batch = train_batch.to(device)
            y_pred = model(train_batch)
            y_true = train_batch.y
            train_loss = F.mse_loss(y_pred, y_true)
            train_loss.backward()
            train_loss_all += BATCH_SIZE * float(train_loss)
            optimizer.step()
            torch.cuda.empty_cache()

        train_loss_all = train_loss_all / len(train_loader.dataset)
        train_losses.append(train_loss_all)

        if epoch % LR_DECAY_STEP == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / LR_DECAY_VALUE
        
    return train_losses, model, test_loader, test_u_indices


# function used to test the trained IGMC model across various metrics like MSE, Precision@k, recall@k and F1 Score
def test(model, test_loader, test_u_indices, topK = 10):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.synchronize()
        device = torch.device('cuda')
    model.eval()
    test_loss = 0
    predictions = None
    true_labels = None
    for index, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        with torch.no_grad():
            y_pred = model(test_batch)
        y_true = test_batch.y
        test_loss += F.mse_loss(y_pred, y_true, reduction='sum')
        if index == 0:
            predictions = y_pred.clone()
            true_labels = y_true.clone()
        else:
            predictions = torch.cat([predictions, y_pred])
            true_labels = torch.cat([true_labels, y_true])
    mse_loss = float(test_loss) / len(test_loader.dataset)
    print('Test MSE loss', mse_loss)
    print('Test RMSE loss', math.sqrt(mse_loss))
    precisionK, recallK, f1Score = precision_recall_calculation(predictions, true_labels, test_u_indices, topK=10, threshold=3.5)
    print(" Precision@k ", precisionK )
    print(" Recall@k ", recallK )
    print(" F1 Score ", f1Score )
    return 