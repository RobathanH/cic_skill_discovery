# Instructions to Manually Set Up Environment

Installing directly from conda_env.yml didn't work for me, so these are the commands I used to manually install a working environment.

`conda create -n urlb python=3.8`
`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
`conda install pip numpy absl-py pyparsing jupyterlab scikit-image`
`pip install termcolor dm_control tb-nightly imageio imageio-ffmpeg hydra-core hydra-submitit-launcher pandas ipdb yapf mujoco_py scikit-learn matplotlib opencv-python wandb moviepy`