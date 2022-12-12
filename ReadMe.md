### A Novel Deep Reinforcement Learning based Framework for Gait Adjustment

This repository is the official implementation of A Novel Deep Reinforcement Learning based Framework for Gait Adjustment.

#### Requirements

> python>=3.6
>
> torch==1.10.0
>
> osim==3.0.11

#### Virtual Env Config

1. Install the virtual environement

> conda create -n opensim-rl -c kidzik opensim python=3.6

2. activate the environment

> conda activate opensim-rl

3. Install dependencies

>pip install osim-rl==3.0.11
>
>conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch

#### Run Project

1. 进入虚拟环境

>conda activate opensim-rl

2. 运行主代码(例如运行SD3算法)

> python main_SD3.py

#### Additional Notes

> The MDP modeling of our work is written in the file "code\env\osim\env\osim.py". Please replace the original file with this file after the virtual environment is installed.  