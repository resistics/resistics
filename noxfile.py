import nox
from nox_poetry import Session
from nox_poetry import session

nox.options.sessions = "pytest", "safety"

python_pytest = ["3.7", "3.8", "3.9"]
python_mypy = ["3.7", "3.8", "3.9"]
locations_mypy = ["resistics"]
python_safety = ["3.7", "3.8", "3.9"]
python_docs = "3.9"


@session(python=python_pytest)
def pytest(session):
    """Run unit tests and doctests"""
    session.install(".")
    session.install(
        session, "matplotlib", "pytest", "coverage[toml]", "pytest-cov", "pytest-html"
    )
    session.run("pytest")


@session(python=python_mypy)
def mypy(session):
    """Perform static type checking"""
    args = session.posargs or locations_mypy
    session.install(".", "mypy")
    session.run("mypy", *args)


@session(python=python_safety)
def safety(session):
    """Check import modules for safety"""
    requirements = session.poetry.export_requirements()
    session.install("safety")
    session.run("safety", "check", "--full-report", f"--file={requirements}")


@session(python=python_docs)
def docs(session: Session) -> None:
    """Build the documentation."""
    from pathlib import Path
    import shutil

    args = session.posargs or ["docs/source", "docs/build"]
    session.install(".")
    session.install("sphinx", "furo")

    build_dir = Path("docs", "build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)
