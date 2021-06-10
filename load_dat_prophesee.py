import numpy as np
from src.io.psee_loader import PSEELoader

event_dtype = np.dtype([('x', np.uint16), ('y', np.uint16), ('ts', np.float32), ('p', np.int8)])


def load_dat_events(filename):
    """
    :param filename:样本的文件名
    :return: 以event_dtype存储的np数组
    """
    file = PSEELoader(filename)
    evs = file.load_n_events(file.event_count())

    array_evs = np.rec.fromarrays([evs['x'], evs['y'], evs['t'], evs['p']], dtype=event_dtype)

    return array_evs