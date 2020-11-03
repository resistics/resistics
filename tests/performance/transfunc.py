from logging import getLogger
from pathlib import Path

logger = getLogger()


def calculate_transfer_function(proj):
    from resistics.project.transfunc import processProject

    processProject(proj, sites=["site1_mt"])


if __name__ == "__main__":
    import cProfile, pstats
    from resistics.project.io import loadProject
    from datapaths import performance_project
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    proj = loadProject(performance_project)
    profiler = cProfile.Profile()
    profiler.enable()
    calculate_transfer_function(proj)
    profiler.disable()
    stats = pstats.Stats(profiler)
    outfile = Path("results_transfunc", f"{now}_transfunc_process.prof")
    stats.dump_stats(str(outfile))
    stats.sort_stats("tottime").print_stats()
