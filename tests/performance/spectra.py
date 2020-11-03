from pathlib import Path
from logging import getLogger

logger = getLogger()


def calculate_spectra(proj):
    from resistics.project.spectra import calculateSpectra

    calculateSpectra(proj, sites=["site1_mt"])


if __name__ == "__main__":
    import cProfile, pstats
    from resistics.project.io import loadProject
    from datapaths import performance_project
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    proj = loadProject(performance_project)
    profiler = cProfile.Profile()
    profiler.enable()
    calculate_spectra(proj)
    profiler.disable()
    stats = pstats.Stats(profiler)
    outfile = Path("results_spectra", f"{now}_calculate_spectra.prof")
    stats.dump_stats(str(outfile))
    stats.sort_stats("tottime").print_stats()
