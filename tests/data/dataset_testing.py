from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List
import os
from loguru import logger
import numpy as np
import pandas as pd

from resistics.config import Configuration
from resistics.time import TimeMetadata, TimeReader, TimeReaderAscii, TimeReaderNumpy
from resistics.time import RemoveMean, InterpolateNans, Multiply
from resistics.decimate import DecimationSetup
from resistics.window import WindowerTarget
from resistics.letsgo import quick_tf

load_dotenv()
data_path = Path(os.getenv("TEST_DATA_PATH_TIME"))
emslab_path = data_path / "emslab"
emslab_csv_path = emslab_path / "emsl01.csv"
ascii_path = data_path / "ascii"
numpy_path = data_path / "numpy"


def write_ascii(time_dict: Dict[str, Any], chans: List[str], df: pd.DataFrame):
    """Write ascii dataset"""
    logger.info("Writing ascii data")
    chans_dict = {x: {"name": x, "data_files": "data.txt"} for x in chans}
    time_dict["chans_metadata"] = chans_dict
    metadata = TimeMetadata(**time_dict)
    metadata.write(ascii_path / "metadata.json")
    df.to_csv(ascii_path / "data.txt", sep="\t", index=False)


def write_numpy(time_dict: Dict[str, Any], chans: List[str], df: pd.DataFrame):
    """Write numpy dataset"""
    logger.info("Writing out numpy data")
    chans_dict = {x: {"name": x, "data_files": "data.npy"} for x in chans}
    time_dict["chans_metadata"] = chans_dict
    metadata = TimeMetadata(**time_dict)
    metadata.write(numpy_path / "metadata.json")
    data = df.values.T.astype(np.float32)
    np.save(numpy_path / "data.npy", data)


def prepare():
    """Prepare the data for writing"""
    fs = 0.05
    chans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
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
    # write out
    write_ascii(dict(time_dict), chans, df)
    write_numpy(dict(time_dict), chans, df)


def process(time_reader: TimeReader, dir_path: Path):
    """Process the data"""
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
    solution = quick_tf(dir_path, config)
    fig = solution.tf.plot(solution.freqs, solution.components)
    fig.show()


if __name__ == "__main__":
    prepare()
    process(TimeReaderNumpy(), numpy_path)
    process(TimeReaderAscii(delimiter="\t", n_header=1), ascii_path)
