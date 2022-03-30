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
        df, x_offsets, y_offsets, c_in=1, scaler = None
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
    freq = '5T'
    data_stamp = np.vstack([feat(pd.to_datetime(df.index.values)) for feat in time_features_from_frequency_str(freq)]).transpose(1, 0).astype(np.float32)
#     data_stamp = np.expand_dims(data_stamp,axis = 0)  
    data_stamp = np.expand_dims(data_stamp,axis = 1).repeat(num_nodes,axis = 1)
    data_embed = np.concatenate((data,data_stamp),axis = 2)
#     data_embed = np.squeeze(data_embed,axis = 0)
    print(data_stamp.shape,data_embed.shape)
    x = []
    y = []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples-max(y_offsets))

    for t in range(min_t,max_t):
        x.append(data_embed[t + x_offsets])
        y.append(data_embed[t + y_offsets])
    x = np.stack(x,axis = 0)
    y = np.stack(y,axis = 0)
    print(x.shape,y.shape)
    return x,y



def generate_train_val_test(args):
    
    seq_length_x,seq_length_y = args.seq_length_x,args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)
    
    x_offsets = np.sort(np.arange(-(seq_length_x - 1),1,1))

    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x,y = generate_graph_seq2seq_io_data(
        df,
        x_offsets = x_offsets,
        y_offsets = y_offsets,
        c_in=args.c_in,

    )

    print('x.shape is:',x.shape,'y.shape is:',y.shape)

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
parser.add_argument('--c_in',type=int ,default=1)

parser.add_argument('--y_start',type=int,default=1)
parser.add_argument('--dow',action='store_true')

args = parser.parse_args()

if os.path.exists(args.output_dir):
    reply = str(input(f'{args.output_dir} exists.Do you want to overwrite it ? (y/n) ')).lower().strip()
    if reply[0] != 'y': exit()
else :
    os.makedirs(args.output_dir)
generate_train_val_test(args)
