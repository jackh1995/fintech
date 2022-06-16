import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from constants import selected_features, selected_features_esun, selected_features_fugle, cat_features_esun, cat_features_fugle

def make_quota(a, b):
    if math.isnan(b):
        return a
    else:
        return min(a, b)

def to_class(x):
    '''
    0~10萬
    10~30萬(不含10萬)
    30~50萬(不含30萬)
    50~100萬(不含50萬)
    '''
    if 0 <= x and x < 1E5:
        return 0
    if 1E5 <= x and x < 3E5:
        return 1
    if 3E5 <= x and x < 5E5:
        return 2
    else:
        return 3
        
def read_data(data_csv='./data/ooa_features_v1.csv', source=None, selected_features=None, cat_features=None):
    assert source in {'FUGLE', 'ESUN'}, 'source is not defined!'
    assert selected_features is not None, 'please select some features!'
    assert cat_features is not None, 'please identify what features are categorical!'

    df_all = pd.read_csv(data_csv)

    # select features
    df_all  = df_all[selected_features]
    df_all = df_all[df_all['occupation'] <= 33]

    # define the label to predict
    df_all['y_num'] = df_all[['quota_now', 'quota_now_elec']].apply(lambda x: make_quota(*x), axis=1)
    df_all = df_all[df_all['y_num']<=1e6]
    df_all['y_cat'] = df_all['y_num'].apply(lambda x: to_class(x))
    df_all = df_all.drop(['quota_now', 'quota_now_elec'], axis=1)

    # drop: isReject
    df_all = df_all[df_all['isReject']==0]
    df_all = df_all.drop('isReject', axis=1)

    # drop source Anue 
    df_all = df_all[df_all['source'] != 'Anue']
    df_all = df_all.replace({"source": {'FUGLE': 0, '玉證': 1}})

    if source == 'FUGLE':
        df_all = df_all[df_all['source'] == 0]
    else:
        df_all = df_all[df_all['source'] == 1]
    df_all = df_all.drop('source', axis=1)

    # take the absolute value of salary to avoid negative values
    df_all['salary'] = df_all['salary'].apply(lambda x: abs(x))

    df_all = df_all.dropna()
    # display(df_all.head())

    # normalization
    df_x_raw = df_all.iloc[:, :-2]
    df_y = df_all.iloc[:, -1]
    num_features = [col for col in df_x_raw.columns if col not in cat_features]
    df_x_num = df_x_raw[num_features].apply(lambda x: x/x.max(), axis=0)
    df_x_cat = df_x_raw[cat_features]
    df_x = pd.concat([df_x_num, df_x_cat], axis=1)
    df_x.reset_index(drop=True, inplace=True)
    df_y.reset_index(drop=True, inplace=True)

    # one-hot encoding the categorical data
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(df_x[cat_features]).toarray())
    df_x = df_x.join(encoder_df)
    df_x.drop(cat_features, axis=1, inplace=True)

    # display(df_x.head())
    # display(df_y.head())

    return df_x, df_y

class MyNetwork(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x

class MyDataset(Dataset):
    def __init__(self, df_x, df_y):
        x = df_x.values
        y = df_y.values

        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(model, train_loader, optimizer, epoch):
    global device
    model.train()
    train_loss = 0
    n_correct = 0
    n_total = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, y_pred = output.max(1)
        n_total += y.size(0)
        n_correct += y_pred.eq(y).sum().item()
        
    train_loss /= len(train_loader)
    train_acc = n_correct / n_total
    print(f'[epoch] {epoch}, [train acc] {train_acc:.2%}, [train loss] {train_loss:.6f}', end=' ')

def valid(model, valid_loader):
    global device
    global best_loss
    global patience
    global n_patience
    global early_stop
    
    model.eval()
    valid_loss = 0
    n_correct = 0
    n_total = 0
    
    y_pred_list = []
    
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            valid_loss += nn.CrossEntropyLoss()(output, y).item()
            _, y_pred = output.max(1)
            n_total += y.size(0)
            n_correct += y_pred.eq(y).sum().item()
            
            y_pred_list.append(y_pred.detach().cpu().numpy())

    valid_loss /= len(valid_loader)
    valid_acc = n_correct / n_total

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'model.ckpt')
        patience = 0
    else:
        patience += 1
        if patience > n_patience:
            early_stop = True
    yp = np.concatenate(y_pred_list)
    yg = valid_loader.dataset.y.numpy()

    precision, recall, fscore, support = score(yg, yp, zero_division=0)
    micro = score(yg, yp, zero_division=0, average='micro')
    macro = score(yg, yp, zero_division=0, average='macro')
    
    print(f'[valid acc] {valid_acc:.2%}, [valid loss] {valid_loss:.6f} [esc] {patience}/{n_patience}')
    
    print(f'{macro = }')
    print(f'{micro = }')
    res_df = pd.DataFrame({
        'precision' : precision,
        'recall' : recall,
        'fscore' : fscore,
        'support' : support
    })
    print(res_df)

if __name__ == '__main__':

    # ---------------------------------------------------------------------------- #
    #                                   CONSTANTS                                  #
    # ---------------------------------------------------------------------------- #
    
    seed = 840519
    test_size = 0.015
    n_epoch = 200
    patience = 0
    n_patience = 10

    # ---------------------------------------------------------------------------- #
    #                                    SOURCE                                    #
    # ---------------------------------------------------------------------------- #

    # source = 'FUGLE'
    source = 'ESUN'
    if source == 'FUGLE':
        selected_features = selected_features_fugle
        cat_features = cat_features_fugle
    else:
        selected_features = selected_features_esun
        cat_features = cat_features_esun

    # ---------------------------------------------------------------------------- #
    #                                  DATALOADER                                  #
    # ---------------------------------------------------------------------------- #
    
    print(f'Loading {source} data...', end='', flush=True)
    df_x, df_y = read_data(data_csv='./data/ooa_features_v1.csv', source=source, selected_features=selected_features, cat_features=cat_features)

    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

    train_dataset = MyDataset(train_x, train_y)
    valid_dataset = MyDataset(test_x, test_y)

    label_counter = train_y.value_counts()
    total_counts = len(train_y)
    for k in label_counter.keys():
        label_counter[k] = total_counts/label_counter[k]
    example_weights = [label_counter[e] for e in train_y]
    sampler = WeightedRandomSampler(example_weights, total_counts)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=64)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    print('done')

    # ---------------------------------------------------------------------------- #
    #                                   TRAINING                                   #
    # ---------------------------------------------------------------------------- #

    n_input = len(df_x.columns)
    model = MyNetwork(n_input)
    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model.to(device)
    optimizer=optim.Adam(model.parameters())
    
    early_stop = False
    best_loss = math.inf
    
    for i in range(n_epoch):
        train(model, train_loader, optimizer, i)
        valid(model, valid_loader)
        print()

        if early_stop:
            print('Early stopped!')
            break