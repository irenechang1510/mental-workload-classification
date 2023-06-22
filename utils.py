import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from STM import STM
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def build_source_classifier(source_df, features, y, target_df):
    md = LogisticRegression().fit(source_df[features], source_df[y])
    target_score = md.score(target_df[features], target_df[y])
    return md, target_score

# return list of accuracy and list of parameters
def tune_n_best_classifiers(n_domains, stm, val_set, top_k):
    list_combinations = [top_k]
    list_acc = []
    # best classifier model
    stm.get_top_k(top_k)
    stm.fit(val_set)
    list_acc.append(stm.get_accuracy())

    # randomized models
    for _ in range(9):
        n_domain_to_draw = np.random.randint(1, n_domains + 1)
        domain_to_draw = random.sample(range(n_domains), n_domain_to_draw)
        while domain_to_draw in list_combinations:
            n_domain_to_draw = np.random.randint(1, n_domains + 1)
            domain_to_draw = random.sample(range(n_domains), n_domain_to_draw)
        list_combinations.append(domain_to_draw)
        stm.get_top_k(domain_to_draw)
        stm.fit(val_set)
        list_acc.append(stm.get_accuracy())
        
    return list_combinations, list_acc

# def tune_Cs(n_domains, list_Cs, val_set, features):
#     list_combinations = []
#     list_acc = []
#     for _ in range(10):
#         C_list = random.choices(list_Cs, k = n_domains)
#         h = STM(num_domains=n_domains, n_class=4, features = features, label = 'label', Cs=1000, 
#         kernels = 'poly', k_list = 1, beta_list = 0.1, gamma_list = 0.1)

def make_target(target):
    t_path = os.path.join('DataSet', 'sub_' + str(target) + '.csv')
    t_df = pd.read_csv(t_path)
    classes = list(t_df.label.unique())
    t_df = t_df.groupby('chunk').agg('mean').reset_index(drop=True)
    return t_df, classes

def make_datasets(sources, target, n = 50, seed = 0):
    # target is coded 0, the next sources are coded 1, ..., n
    t_df, classes = make_target(target)
    
    num_target_train = int(len(t_df)/2)
    t_df_train, t_df_test = t_df.iloc[:num_target_train, :], t_df.iloc[num_target_train:, :]
    
    # make sure all classes are present in both sets
    # assert len(t_df_train['label'].unique()) == len(classes)
    # assert len(t_df_test['label'].unique()) == len(classes)

    t_df_train['source_no'] = [0] * len(t_df_train)

    source_dfs = pd.DataFrame([], columns = t_df.columns)
    for id, s in enumerate(sources):
        # print("Process data for source subject ", s)
        s_path = os.path.join('DataSet', 'sub_' + str(s) + '.csv')
        df = pd.read_csv(s_path)
        
        # group by and taking the mean
        df = df.groupby('chunk').agg('mean').reset_index(drop=True)
        df = select_data(df, n, classes, seed = seed)
        
        # for purpose of fitting model
        df['source_no'] = (id+1)
        
        source_dfs = source_dfs.append(df, ignore_index=True)

    train_dfs = pd.concat([t_df_train, source_dfs], ignore_index=True)
    return (train_dfs.drop('label', axis=1), train_dfs['label']), (t_df_test.drop('label', axis=1), t_df_test['label'])
    
def select_data(subj, n, classes, seed):
    if n == 1:
        return subj
    elif n < 1:
        n = int(len(subj) * n)

    # select n instances randomly from each class
    new_df = []
    for c in classes:
        new_df.append(subj[subj['label'] == c].sample(n,random_state=seed))

    return pd.concat(new_df, ignore_index=True)

def viz_train_test_split(train, test):
    t23_train_viz = train[['chunk']]
    t23_train_viz.loc[:, 'color'] = 'green'
    t23_test_viz = test[['chunk']]
    t23_test_viz.loc[:, 'color'] = 'yellow'
    t23_viz = pd.concat([t23_train_viz, t23_test_viz], ignore_index=True)
    index_df = pd.DataFrame(np.arange(1, 1489), columns = ['chunk'])
    t23_viz = t23_viz.merge(index_df, how='right', on='chunk')
    t23_viz['color'] = t23_viz['color'].fillna('grey')

    status = t23_viz.color.values
    t23_viz['dummy_x'] = ['X'] * len(t23_viz)
    t23_viz['dummy_y'] = [1] * len(t23_viz)
    t23_piv = t23_viz.pivot(index='dummy_x', columns='chunk', values='dummy_y')
    ax = t23_piv.plot(
        kind='barh', stacked=True, color = list(status), legend = False, figsize=(10, 3), width = 0.3)
    

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def splitFolds(X_train_dfs, n = 5):
    # splitting into a default of 5 folds
    X = X_train_dfs.copy()
    X = X.sample(frac=1, random_state=0)
    # print(X)
    keys = X_train_dfs.source_no.astype('int').unique()
    domains = dict.fromkeys(keys, None)
    res = dict.fromkeys(range(n), [])
    for key in domains:
        to_subset = X[X.source_no == key]
        # print("to subset of ", key, " includes: ", list(to_subset.index))
        # list of iterable
        to_subset_indices = list(split(to_subset.index, n))
        # print(to_subset_indices)
        
        for i in res:
            res[i] =  res[i] + list(to_subset_indices[i])
    return list(res.values())

# def train_test_split(X_train_dfs, cv = 5):
#     train_indices = []
#     test_indices = []
#     folds = splitFolds(X_train_dfs, n = cv)
#     for test_split in folds:
#         train_split = list(set(X_train_dfs.index) - set(test_split))
#         train_indices.append(train_split)
#         test_indices.append(test_split)
    
#     return zip(train_indices, test_indices)

def train_test_split(X_train_dfs):
    target_set = X_train_dfs[X_train_dfs.source_no == 0]
    source_set = X_train_dfs.drop(target_set.index)
    train_indices = list(source_set.index)

    num_target_train = int(len(target_set)/2)
    t_df_train, t_df_test = target_set.iloc[:num_target_train, :], target_set.iloc[num_target_train:, :]
    # print(t_df_train.tail(10))
    # print(t_df_test.head(10))
    train_indices.extend(list(t_df_train.index))
    test_indices = list(t_df_test.index)

    return (train_indices, test_indices)