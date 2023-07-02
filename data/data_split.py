import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import KFold

col_num = [('MovieID', 4000),
           ('UserID', 6050),
           ('Age', 7),
           ('Gender', 2),
           ('Occupation', 21),
           ('Year', 83),
           ('Genres', 25),
           ('Title', 5000)]


def read_pkl(file):
    with open(file, 'rb') as f:
        t = pickle.load(f)
    return t


def save_pkl(df, filename):
    with open(filename, 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    big_main = read_pkl('./old/big_train_main.pkl')
    test_test = read_pkl('./old/test_test.pkl')
    whole_set = pd.concat([big_main, test_test])
    # 划分电影集合为movie_set1\2
    movie_set = whole_set['MovieID'].unique()
    np.random.shuffle(movie_set)
    movie_set1 = movie_set[:len(movie_set) // 2]
    movie_set2 = movie_set[len(movie_set) // 2:]
    #划分两个域数据集
    dataset1 = big_main.loc[big_main['MovieID'].isin(movie_set1)]
    dataset2 = big_main.loc[big_main['MovieID'].isin(movie_set2)]
    dataset1 = dataset1.reset_index(drop=True)
    dataset2 = dataset2.reset_index(drop=True)
    save_pkl(dataset1, './dataset_sys1.pkl')

    #处理冷启动域数据集
    userid_counts = dataset2['UserID'].value_counts()
    userid_counts_over_100 = userid_counts.loc[userid_counts > 100]
    old_userid_set = userid_counts_over_100.index.unique()
    old_dataset = dataset2.loc[dataset2['UserID'].isin(old_userid_set)]
    save_pkl(old_dataset, './newdataset/big_train_main.pkl')

    # userid_counts_20_100 = userid_counts.loc[(userid_counts <= 100) & (userid_counts > 20)]
    # new_userid_set = userid_counts_20_100.index.unique()
    # new_dataset = dataset2.loc[dataset2['UserID'].isin(new_userid_set)]

    userid_counts_40_100 = userid_counts.loc[(userid_counts <= 100) & (userid_counts > 40)]
    new_userid_set = userid_counts_40_100.index.unique()
    new_dataset = dataset2.loc[dataset2['UserID'].isin(new_userid_set)]

    # 交叉验证数据集
    old_dataset1 = old_dataset[['UserID', 'MovieID']].sort_values(by='UserID')
    train_a, train_b, train_c, train_d = [], [], [], []
    for user_id, group in old_dataset1.groupby('UserID'):
        num_records = len(group)
        # 如果记录数量不被4整除，则舍弃一些记录
        if num_records % 4 != 0:
            num_to_drop = num_records % 4
            group = group.drop(group.tail(num_to_drop).index)
        group_indices = np.arange(len(group))
        group_indices = np.array_split(group_indices, 4)
        for i in range(4):
            indices = group_indices[i]
            dataset = [group.iloc[idx] for idx in indices]
            if i == 0:
                train_a.extend(dataset)
            elif i == 1:
                train_b.extend(dataset)
            elif i == 2:
                train_c.extend(dataset)
            elif i == 3:
                train_d.extend(dataset)

    # 将数据集转换为dataframe
    train_set_a = pd.DataFrame(train_a)
    train_set_b = pd.DataFrame(train_b)
    train_set_c = pd.DataFrame(train_c)
    train_set_d = pd.DataFrame(train_d)

    train_set_a = train_set_a.merge(old_dataset, on=['UserID', 'MovieID'], how='left')
    train_set_b = train_set_b.merge(old_dataset, on=['UserID', 'MovieID'], how='left')
    train_set_c = train_set_c.merge(old_dataset, on=['UserID', 'MovieID'], how='left')
    train_set_d = train_set_d.merge(old_dataset, on=['UserID', 'MovieID'], how='left')
    save_pkl(train_set_a, './train_oneshot_a.pkl')
    save_pkl(train_set_b, './train_oneshot_b.pkl')
    save_pkl(train_set_c, './train_oneshot_c.pkl')
    save_pkl(train_set_d, './train_oneshot_d.pkl')

    # test数据集处理
    new_dataset1 = new_dataset[['UserID', 'MovieID']]
    grouped = new_dataset1.groupby('UserID')

    test_set1, test_set2, test_set3, test_set4 = [], [], [], []

    for user_id, group in grouped:
        num_records = len(group)
        indices = np.arange(num_records)
        np.random.shuffle(indices)
        for i in range(3):
            # dataset = [group.iloc[idx] for idx in indices[i * 5:(i + 1) * 5]]
            dataset = [group.iloc[idx] for idx in indices[i * 10:(i + 1) * 10]]
            if i == 0:
                test_set1.extend(dataset)
            elif i == 1:
                test_set2.extend(dataset)
            elif i == 2:
                test_set3.extend(dataset)
        dataset = [group.iloc[idx] for idx in indices[30:]]
        test_set4.extend(dataset)

    test_set_a = pd.DataFrame(test_set1)
    test_set_b = pd.DataFrame(test_set2)
    test_set_c = pd.DataFrame(test_set3)
    test_set_d = pd.DataFrame(test_set4)
    test_set_a = test_set_a.merge(new_dataset, on=['UserID', 'MovieID'], how='left')
    test_set_b = test_set_b.merge(new_dataset, on=['UserID', 'MovieID'], how='left')
    test_set_c = test_set_c.merge(new_dataset, on=['UserID', 'MovieID'], how='left')
    test_set_d = test_set_d.merge(new_dataset, on=['UserID', 'MovieID'], how='left')
    save_pkl(test_set_a, './test_oneshot_a.pkl')
    save_pkl(test_set_b, './test_oneshot_b.pkl')
    save_pkl(test_set_c, './test_oneshot_c.pkl')
    save_pkl(test_set_d, './test_test.pkl')