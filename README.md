# OCRL_project_treemanipulate

# installation setup
1. Install IsaacGym from [Nvidia](https://developer.nvidia.com/isaac-gym)
2. Follow the instructions in install.html doc (following these instructions should setup isaacgym directory like shown in this repo)
```
cd /home/mark/course/16745_orcl/OCRL_project_treemanipulate/isaacgym/docs
```

2a. This means creating a new conda environment by running the shell script. Make changes to the .sh and .yml file to rename the conda environment to OCRL or whatever you'd like.
```
./create_conda_env_rlgpu.sh
```

3. Get the isaac-gym-utils (either by git clone or copy & pasting existing)
```
pip install -e isaacgym-utils
```
