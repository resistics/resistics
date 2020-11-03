from logging import getLogger
from pathlib import Path

logger = getLogger()


def calculate_statistics(proj):
    from resistics.project.statistics import calculateStatistics

    calculateStatistics(proj, sites=["site1_mt"])


if __name__ == "__main__":
    import cProfile, pstats
    from resistics.project.io import loadProject
    from datapaths import performance_project
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    proj = loadProject(performance_project)
    # define the statistics to calculate
    profiler = cProfile.Profile()
    profiler.enable()
    calculate_statistics(proj)
    profiler.disable()
    stats = pstats.Stats(profiler)
    outfile = Path("results_transfunc", f"{now}_calculate_statistics.prof")
    stats.dump_stats(str(outfile))
    stats.sort_stats("tottime").print_stats()
