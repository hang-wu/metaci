# Meta Learning-based Causal Graph Learning

Status: Manuscript under review, codebase released

This is the codebase for the manuscript *Developing A Novel Causal Inference Algorithm for Personalized Biomedical Causal Graph Learning Using Meta Machine Learning* by Hang Wu, Wenqi Shi, and May Dongmei Wang.

Below we include instructions for running the algorithms proposed in the manuscript.

## Installation
Complete the following steps:
1. Clone the repo to your desired locations
2. Create a virtual python environment using a environment management software of your choice (e.g. conda, venv)
3. Install the prerequistes via `pip install -r requirements.txt`

Note: this code is tested under python3.9 and pytorch build 2.2.2 with CUDA11.8. This specific version could be installed via command `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.

## Usage
To run a model, simply run `python main.py` and you will obtain relevant code outputs.

The dataset generation and loading scripts can be found in the data folder. And the SACHS dataset can be downloaded in the [link](https://github.com/snarles/causal/blob/master/bnlearn_files/sachs.data.txt).

## Other

We thank the authors of the paper [*DAGs with NO TEARS: Continuous Optimization for Structure Learning*](https://arxiv.org/abs/1803.01422) for releasing their codes on [GitHub](https://github.com/xunzheng/notears), and for simplicity of usage, we import their codebases into our folder structure as-is and build on top of it.