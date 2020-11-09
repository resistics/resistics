from resistics.project.data import ProjectData


class ProjectProcessor():

    def __init__(self, project: ProjectData):
        self.project = project
        self._sites: List[str] = project.get_sites()
        self._fs: List[float] = project.get_fs()
        self._steps: List[Calculator] = []

    def add_step(self, calculator: Calculator) -> None:
        self._step.append(calculator)

    def process(self):
        
