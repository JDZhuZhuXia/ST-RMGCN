import argparse
import numpy as np
import os
import pandas as pd
import psutil
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, scaler = None
) :
    """
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    """
    

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df , axis = -1).astype(np.float32)
#     data = np.expand_dims(data, axis=0).astype(np.float32)
    
#     time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(5, "m")
#     time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))#title 扩张
#     time_in_day = np.expand_dims(time_in_day,axis=0)
    freq = '5T'
    data_stamp = np.vstack([feat(pd.to_datetime(df.index.values)) for feat in time_features_from_frequency_str(freq)]).transpose(1, 0).astype(np.float32)
#     embed = DataEmbedding(c_in, d_model)
    
#加入时间特征5
#     data_stamp = np.expand_dims(data_stamp,axis = 0)  
    data_stamp = np.expand_dims(data_stamp,axis = 1).repeat(207,axis = 1)

#加入统计特征5 每周 各时间片的统计值
    df = np.array(df).reshape(num_samples//288,288,num_nodes)
    df = df.reshape(num_samples//288//7,7,288,num_nodes,1)
    
#     statistic_data = np.array([i*288 for i in range(df.shape[0]//288)])
#     for i in range(288):
#         day_data.append(df[i + statistic_data])
    
    data_mean = np.expand_dims(np.mean(df,1),1)
    data_max = np.expand_dims(np.max(df,1),1)
    data_min = np.expand_dims(np.min(df,1),1)
    data_std = np.expand_dims(np.std(df,1),1)
    data_median = np.expand_dims(np.median(df,1),1)
    day_data = np.concatenate((data_mean,data_max,data_min,data_median,data_std),axis=4) #shape = 17*1*288*207*5   
    day_data = np.tile(day_data,(1,7,1,1,1)).astype(np.float32).reshape(num_samples,num_nodes,5)
    
    
#     data_embed = embed(data,data_stamp)
    print(data_stamp.shape,data.shape,day_data.shape)
    data_embed = np.concatenate((data,day_data,data_stamp),axis = 2)
#     data_embed = np.squeeze(data_embed,axis = 0)

    x = []
    y = []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples-max(y_offsets))

    for t in range(min_t,max_t):
        x.append(data_embed[t + x_offsets])
        y.append(data_embed[t + y_offsets])
    
    mem = psutil.virtual_memory()
    ysy = float(mem.used)/1024/1024/1024
    print('检查点2已使用内存：',ysy)
    x = np.stack(x,axis = 0)
    y = np.stack(y,axis = 0)
    return x,y



def generate_train_val_test(args):
    seq_length_x,seq_length_y = args.seq_length_x,args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)#加载数据y
    x_offsets = np.sort(np.arange(-(seq_length_x - 1),1,1))#concatenate多个数组拼接

    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    x,y = generate_graph_seq2seq_io_data(
        df,
        x_offsets = x_offsets,
        y_offsets = y_offsets,
    )

    print('x.shape is:',x.shape,'y.shape is:',y.shape)

    #数据分为三份存储
    num_samples = len(x)
    num_test = round(num_samples*0.2)
    num_train = round(num_samples*0.7)
    num_val = num_samples - num_train - num_test
    x_train,y_train = x[:num_train],y[:num_train]
    x_test,y_test = x[num_train:num_train+num_test],y[num_train:num_train+num_test]
    x_val ,y_val = x[-num_val:],y[-num_val:]
    for cat in ["train","test","val"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',type = str,default='Data/METR-LA')
parser.add_argument('--traffic_df_filename',type = str,default='Data/metr-la.h5')
parser.add_argument('--seq_length_x',type = int,default=12)
parser.add_argument('--seq_length_y',type=int ,default=12)
# parser.add_argument('--c_in',type=int ,default=1)
# parser.add_argument('--d_model',type=int ,default=8)
parser.add_argument('--y_start',type=int,default=1)

args = parser.parse_args()

if os.path.exists(args.output_dir):
    reply = str(input(f'{args.output_dir} exists.Do you want to overwrite it ? (y/n) ')).lower().strip()#strip()去除空格或指定字符
    if reply[0] != 'y': exit()
else :
    os.makedirs(args.output_dir)
generate_train_val_test(args)

#生成训练数据 维度为 （23974, 12, 207, 2）