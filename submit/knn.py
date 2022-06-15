import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
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
    if x < 1E5:
        return 0
    if 1E5 <= x and x < 3E5:
        return 1
    if 3E5 <= x and x < 5E5:
        return 2
    else:
        return 3

# feature exploration
def plot_corr(df: pd.DataFrame) -> None:
    f = plt.figure(figsize=(10, 8))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=10)
    plt.show()
    plt.close()

def get_weights(df_x: pd.DataFrame, df_y: pd.DataFrame):
    '''Get the chi2 scores for each features in df_x with respect to df_y'''
    x_values = df_x.values
    y_values = df_y.values
    top_k = len(df_x.columns)
    bestfeatures = SelectKBest(score_func=chi2, k=top_k)
    fit = bestfeatures.fit(x_values, y_values)
    df_scores = pd.DataFrame(fit.scores_)

    # visualization
    # df_columns = pd.DataFrame(df_x.columns)
    # featureScores = pd.concat([df_columns, df_scores],axis=1)
    # featureScores.columns = ['Specs','Score']  # naming the dataframe columns
    # print(featureScores.nlargest(top_k, 'Score'))  # print 10 best feature
    
    return np.log(df_scores[0].values)

def get_distance(x: np.array, y: np.array, num_features, cat_features, weights=None) -> float:
    """Compute the distance between the instance x and y (numpy arrays)."""

    n_num = len(num_features)
    n_cat = len(cat_features)

    res = 0

    if weights is not None:
        for i in range(n_num):
            res += (float(x[i]) - float(y[i]))**2 * weights[i]

        for i in range(n_num, n_num+n_cat):
            if x[i] != y[i]:
                res += weights[i]
    else:
        for i in range(n_num):
            res += (float(x[i]) - float(y[i]))**2

        for i in range(n_num, n_num+n_cat):
            if x[i] != y[i]:
                res += 1

    return res

def predict(test_x, num, nbrs):
    global train_y
    pred_indices = nbrs.kneighbors(test_x.iloc[:num])
    pred_y = [train_y.iloc[x].values for x in pred_indices[1]]
    return stats.mode(pred_y, axis=1).mode.squeeze()


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
    df_all = df_all[df_all['quota_now']<=1e6]
    df_all['y_cat'] = df_all['quota_now'].apply(lambda x: to_class(x))
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

    # take the absolute value of salary to avoid negative values (values being negative is a bug)
    df_all['salary'] = df_all['salary'].apply(lambda x: abs(x))

    df_all = df_all.dropna()
    # display(df_all.head())

    # normalization
    df_x_raw = df_all.iloc[:, :-2]
    df_y = df_all.iloc[:, -1]
    # cat_features = ['source', 'occupation', 'hasOtherComAccount', 'lead_job_id']
    # cat_features = ['occupation', 'hasOtherComAccount', 'lead_job_id']
    cat_features = ['occupation', 'hasOtherComAccount']
    num_features = [col for col in df_x_raw.columns if col not in cat_features]
    df_x_num = df_x_raw[num_features].apply(lambda x: x/x.max(), axis=0)
    df_x_cat = df_x_raw[cat_features]
    df_x = pd.concat([df_x_num, df_x_cat], axis=1)
    # display(df_x.head())
    # display(df_y.head())
    # df_x.info()

    return df_x, df_y

if __name__ == '__main__':
    
    # knn constants
    n_neighbors = 2
    seed = 840519
    test_size = 0.015
    
    # source = 'FUGLE'
    source = 'ESUN'
    if source == 'FUGLE':
        selected_features = selected_features_fugle
        cat_features = cat_features_fugle
    else:
        selected_features = selected_features_esun
        cat_features = cat_features_esun

    print(f'Loading {source} data...', end='', flush=True)
    df_x, df_y = read_data(data_csv='./data/ooa_features_v1.csv', source=source, selected_features=selected_features, cat_features=cat_features)

    num_features = [col for col in df_x.columns if col not in cat_features]

    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=test_size, random_state=seed)
    print('done')

    # build model
    print(f'Building KNN model using {len(train_y)} intances ({n_neighbors=}, {seed=})...', end='', flush=True)
    weights = get_weights(train_x, train_y)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=get_distance, metric_params={'weights': weights, 'num_features': num_features, 'cat_features': cat_features})
    print('done')

    # train model
    print('Training...', end='', flush=True)
    nbrs.fit(train_x)
    print('done')

    # testing
    n_test = len(test_y)
    print(f'Testing {n_test} instances...', end='', flush=True)
    pred = predict(test_x, n_test, nbrs)
    print('done')
    gt = list(test_y[:n_test].values)
    precision, recall, fscore, support = score(gt, pred)
    res_df = pd.DataFrame({
        'precision' : precision,
        'recall' : recall,
        'fscore' : fscore,
        'support' : support
    })
    
    print(res_df)