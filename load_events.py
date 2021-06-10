import numpy as np
from events_tfds.data_io.aedat import *
from events_tfds.data_io import dvs

HEX = 16
event_dtype = np.dtype([('x', np.uint16), ('y', np.uint16), ('ts', np.float32), ('p', np.int8)])


def load_bin_events(filename, polarities, to_secs):
    """
        Reads a binary file containing events. To use with
        the N-MNIST or N-CALTECH101 dataset

        Each event occupies 40 bits as described below:
            bit 39 - 32: Xaddress (in pixels)
            bit 31 - 24: Yaddress (in pixels)
            bit 23: Polarity (0 for OFF, 1 for ON)
            bit 22 - 0: Timestamp (in microseconds)

    """

    # 5 Named bytes
    orchard_event_dtype = [('x', np.uint8), ('y', np.uint8), ('tp1', np.uint8),
                           ('tp2', np.uint8), ('tp3', np.uint8)]

    # Open file in binary mode
    fp = open(filename, "rb")

    orchard_events = np.fromfile(fp, dtype=orchard_event_dtype, count=-1)

    x = orchard_events['x']
    y = orchard_events['y']

    p = np.right_shift(np.bitwise_and(orchard_events['tp1'].astype(np.uint32),
                                      int('80', HEX)), 7)
    ts1 = np.left_shift(np.bitwise_and(orchard_events['tp1'].astype(np.uint32),
                                       int('7F', HEX)), 16)
    ts2 = np.left_shift(np.bitwise_and(orchard_events['tp2'].astype(np.uint32),
                                       int('FF', HEX)), 8)
    ts3 = np.left_shift(orchard_events['tp3'].astype(np.uint32), 0)

    ts = np.bitwise_or(np.bitwise_or(ts1, ts2), ts3)

    # Transform from millis to secs
    if to_secs:
        ts = ts / 1e6

    # transform to -1,1
    if polarities[0] == -1:
        p = -1 + 2 * p

    recarray = np.rec.fromarrays([x, y, ts, p], dtype=event_dtype)

    return recarray


def load_aedat_events(filename):
    """
    用 https://github.com/jackd/events-tfds 的方法读取aedat格式的事件样本
    """
    with open(filename, 'rb') as fp:
        t, x, y, polarity = dvs.load_events(fp)
        x = 127 - x
        polarity = np.logical_not(polarity)
        polarity = polarity + 0
        t = t/1e6
        array_evs = np.rec.fromarrays([x, y, t, polarity], dtype=event_dtype)
    return array_evs