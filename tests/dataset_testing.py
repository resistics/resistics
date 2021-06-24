from dotenv import load_dotenv
from pathlib import Path
import os
from loguru import logger
import numpy as np
import pandas as pd

from resistics.config import Configuration
from resistics.time import TimeMetadata, TimeReaderNumpy
from resistics.time import RemoveMean, InterpolateNans, Multiply
from resistics.decimate import DecimationSetup
from resistics.window import WindowerTarget
from resistics.letsgo import quick_read, quick_tf

load_dotenv()
data_path = Path(os.getenv("TEST_DATA_PATH_TIME"))
emslab_path = data_path / "emslab"
emslab_csv_path = emslab_path / "emsl01.csv"
ascii_path = data_path / "ascii"
numpy_path = data_path / "numpy"


def write_ascii(metadata: TimeMetadata, df: pd.DataFrame):
    """Write ascii dataset"""
    logger.info("Writing ascii data")
    pass


def write_numpy(metadata: TimeMetadata, df: pd.DataFrame):
    """Write numpy dataset"""
    logger.info("Writing out numpy data")
    metadata.write(numpy_path / "metadata.json")
    data = df.values.T.astype(np.float32)
    np.save(numpy_path / "data.npy", data)


def prepare():
    """Prepare the data for writing"""
    fs = 0.05
    chans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
    chan_to_file = {
        "Ex": "data.npy",
        "Ey": "data.npy",
        "Hx": "data.npy",
        "Hy": "data.npy",
        "Hz": "data.npy",
    }

    logger.info("Reading data")
    df = pd.read_csv(emslab_csv_path, index_col=0, parse_dates=True)
    df = df.interpolate().ffill().bfill()
    logger.info("Preparing metadata")
    time_dict = {
        "fs": fs,
        "chans": chans,
        "n_chans": len(chans),
        "n_samples": len(df.index),
        "first_time": df.index[0],
        "last_time": df.index[-1],
    }
    chans_dict = {}
    for chan in chans:
        chans_dict[chan] = {
            "name": chan,
            "data_files": chan_to_file[chan],
        }
    time_dict["chans_metadata"] = chans_dict
    # write out
    metadata = TimeMetadata(**time_dict)
    write_ascii(metadata, df)
    write_numpy(metadata, df)


def process(time_reader):
    dec_setup = DecimationSetup(n_levels=5, per_level=5)
    windower = WindowerTarget(target=3000)
    time_processors = [
        InterpolateNans(),
        RemoveMean(),
        Multiply(multiplier={"Hx": -1, "Hy": -1, "Hz": -1}),
    ]
    config = Configuration(
        name="testing",
        time_readers=[time_reader],
        time_processors=time_processors,
        dec_setup=dec_setup,
        windower=windower,
    )
    # fig = quick_view(numpy_path)
    time_data = quick_read(config, numpy_path)
    time_data.metadata.summary()
    solution = quick_tf(numpy_path, config)
    fig = solution.tf.plot(solution.freqs, solution.components)
    fig.show()


if __name__ == "__main__":
    prepare()
    process(TimeReaderNumpy())
