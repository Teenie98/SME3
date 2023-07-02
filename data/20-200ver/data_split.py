import numpy as np
import pickle
import pandas as pd

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


def split_dataset(df):
    # 按照用户进行分组
    grouped = df.groupby('UserID')
    user_list = list(grouped.groups.keys())

    # 随机选取一些电影，并将这些电影从原始数据集中移除
    df1 = pd.DataFrame(columns=df.columns)
    df2 = pd.DataFrame(columns=df.columns)
    for user in user_list:
        user_df = grouped.get_group(user)
        n = len(user_df)
        split_idx = int(n * 0.5)
        user_df1 = user_df.sample(n=split_idx)
        user_df2 = user_df.drop(user_df1.index)
        df1 = pd.concat([df1, user_df1])
        df2 = pd.concat([df2, user_df2])

    # 重置索引以确保连续性
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    return df1, df2


def split_warmupdata(df):
    # 按照 MovieID 进行分组
    grouped = df.groupby('MovieID')
    movieid_list = list(grouped.groups.keys())

    # 随机选取每个 MovieID 的 20 条信息，并将选取的信息存储到列表中
    data_list = []
    for movieid in movieid_list:
        movieid_df = grouped.get_group(movieid)
        movieid_sample = movieid_df.sample(n=20, random_state=42)
        data_list.extend(movieid_sample.to_dict('records'))

    # 将列表中的信息分配到 4 个数据集中，每个数据集包含每个 MovieID 的 20 条信息
    np.random.shuffle(data_list)
    n = len(data_list)
    df1 = pd.DataFrame(data_list[:n // 4]).sort_values(by=['MovieID'])
    df2 = pd.DataFrame(data_list[n // 4:n // 2])
    df3 = pd.DataFrame(data_list[n // 2:3 * n // 4])
    df4 = pd.DataFrame(data_list[3 * n // 4:])

    return df1, df2, df3, df4


def save_pkl(df, filename):
    with open(filename, 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    big_main = read_pkl('./old/big_train_main.pkl')

    data1, data2 = split_dataset(big_main)
    save_pkl(data1, './dataset_sys1.pkl')
    save_pkl(data2, './big_train_main.pkl')

    df1, df2, df3, df4 = split_warmupdata(data2)
    save_pkl(df1, './train_oneshot_a.pkl')
    save_pkl(df2, './train_oneshot_b.pkl')
    save_pkl(df3, './train_oneshot_c.pkl')
    save_pkl(df4, './train_oneshot_d.pkl')
