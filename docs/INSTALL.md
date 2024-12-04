# Installation

This doc provides instructions to setup the environment.

## Install CARLA
* Download and unzip [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15)

## Install dependencies

### 1 Setup environment for CARLA
* First, inside the repo, create a dedicated conda environment. Refer [here](https://www.anaconda.com/products/individual#Downloads) if you do not have conda.

```
conda env create -f environment.yaml
```
* Inside the conda environment, install the CARLA PythonAPI `easy_install [PATH TO CARLA EGG]`. Refer to [this link](https://leaderboard.carla.org/get_started/) if you are confused at this step.

### 2 Installation for Scene Understanding Module and Heuristic Process
* Setup enviroment for Scene Understanding Module (Qwen-VL). Refer to [this link](https://github.com/QwenLM/Qwen-VL) if you are confused at this step.

```
conda create -n Qwen-VL python=3.9
pip install -r tools/requirements_vlm.txt
```
* Setup enviroment for Heuristic Process (Qwen1.5). Refer to [this link](https://github.com/QwenLM/Qwen2.5/tree/v1.5) if you are confused at this step.

```
conda create -n Qwen1.5 python=3.9
pip install -r tools/requirements_llm.txt
```

## Configure environment variables

Set the following environmental variables in [eval_leapad.sh](../leaderboard/scripts/eval_leapad.sh). 

```bash
#!/bin/bash

export CARLA_ROOT=[LINK TO YOUR CARLA FOLDER]
export TEAM_AGENT=team_code/leapad_agent.py
export CHECKPOINT_ENDPOINT=result/test.json
export ROUTES=${LEADERBOARD_ROOT}/data/routes_testing.xml # you can change the path to xml file for different routes 
```

