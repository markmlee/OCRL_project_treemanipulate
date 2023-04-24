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

# verifying correct installation
There are two examples you can run to verify correct installation. One for isaacgym, and one for isaacgym-utils.

1. run the franka_attractor.py in isaacgym
```
cd /home/mark/course/16745_orcl/OCRL_project_treemanipulate/isaacgym/python/examples
python  franka_attractor.py
```

2. run the franka_tree.py in isaacgym-utils
```
cd /home/mark/course/16745_orcl/OCRL_project_treemanipulate/isaacgym-utils-ocrl/examples
python franka_tree.py 
```

# Running script to generate tree
This script creates the tree and visualizes in IsaacGym sim 
```
cd /home/mark/course/16745_orcl/OCRL_project_treemanipulate/isaacgym-utils-ocrl/scripts
python generate_tree_files.py 
```

![loaded viz](doc/IG_tree.png)
