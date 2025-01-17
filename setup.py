from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

setup(
    name="transformers_learn_mdp",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements() + ["mcts@git+https://github.com/metric-space/mcts.git"]
)
