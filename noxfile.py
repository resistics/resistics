import nox

nox.options.sessions = "pytest", "safety"

# pytest options
python_pytest = ["3.7", "3.8", "3.9"]
# mypy
python_mypy = ["3.7", "3.8", "3.9"]
locations_mypy = ["resistics"]
# linting options
python_lint = ["3.7", "3.8", "3.9"]
locations_lint = "resistics", "noxfile.py"
# safety options
python_safety = ["3.7", "3.8", "3.9"]
# black analysis options
python_black = ["3.9"]
locations_black = "resistics", "noxfile.py"


def install_with_constraints(session, *args, **kwargs):
    """Install packages with constraints"""
    session.run(
        "poetry",
        "export",
        "--dev",
        "--format=requirements.txt",
        "--without-hashes",
        "--output=requirements_nox.txt",
        external=True,
    )
    session.install("--constraint=requirements_nox.txt", *args, **kwargs)


@nox.session(python=python_pytest)
def pytest(session):
    """Run unit tests and doctests"""
    session.install(".")
    install_with_constraints(
        session, "matplotlib", "pytest", "coverage[toml]", "pytest-cov", "pytest-html"
    )
    session.run("pytest")


@nox.session(python=python_mypy)
def mypy(session):
    """Perform static type checking"""
    session.install(".")
    args = session.posargs or locations_mypy
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python=python_lint)
def lint(session):
    """Perform code quality analysis"""
    args = session.posargs or locations_lint
    install_with_constraints(
        session, "flake8", "flake8-black", "flake8-bugbear", "flake8-bandit"
    )
    session.run("flake8", *args)


@nox.session(python=python_safety)
def safety(session):
    """Check import modules for safety"""
    session.run(
        "poetry",
        "export",
        "--dev",
        "--format=requirements.txt",
        "--without-hashes",
        f"--output=requirements_safety.txt",
        external=True,
    )
    session.install("--constraint=requirements_safety.txt", "safety")
    session.run("safety", "check", "--file=requirements_safety.txt", "--full-report")


@nox.session
def darglint(session):
    """Perform docstring checks"""
    install_with_constraints(session, "darglint")
    session.run("darglint", "resistics/")


@nox.session(python=python_black)
def black(session):
    """Format with black - this will change files"""
    args = session.posargs or locations_black
    install_with_constraints(session, "black")
    session.run("black", *args)
