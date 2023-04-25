import argparse

import numpy as np
from numpy import save 
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.assets import GymTree
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np, quat_to_np
from isaacgym_utils.policy import GraspBlockPolicy, MoveBlockPolicy, GraspTreePolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera, draw_spheres
import pdb
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_tree.yaml')
    parser.add_argument('--cfg', '-c', type=str, default='/home/mark/github/isaacgym-utils/cfg/franka_tree_force_ocrl.yaml')


    
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)


    scene = GymScene(cfg['scene'])


    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    tree = GymTree(cfg['tree'], scene, actuation_mode='joints')

    # block = GymBoxAsset(scene, **cfg['block']['dims'],  shape_props=cfg['block']['shape_props'])


    franka_transform = gymapi.Transform(p=gymapi.Vec3(-0.5, 0, 0))
    tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))

    franka_name, tree_name, block_name = 'franka', 'tree', 'block'

    current_iteration = 0
    num_iteration = 100
    force_magnitude = 50
    push_toggle = True
    
    global vertex_init_pos, vertex_final_pos, force_applied 

    vertex_init_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    vertex_final_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    force_applied = np.zeros((3,tree.num_links)) #fx,fy,fz     


    


    def setup(scene, _):

        scene.add_asset(franka_name, franka, franka_transform, collision_filter=1) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions
        # scene.add_asset('block', block, gymapi.Transform(p=gymapi.Vec3(-1, -1, cfg['block']['dims']['sz']/2)) )

    scene.setup_all_envs(setup)    

    
    no_contact = True
    force = np_to_vec3([0, 0, 0])

    def get_link_poses():
        vertex_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw

        for i in range(tree.num_links):
            link_tf = tree.get_link_transform(0, tree_name, tree.link_names[i])
            pos = vec3_to_np(link_tf.p)
            quat = quat_to_np(link_tf.r)

            vertex_pos[0,i] = pos[0]
            vertex_pos[1,i] = pos[1]
            vertex_pos[2,i] = pos[2]
            vertex_pos[3,i] = quat[0]
            vertex_pos[4,i] = quat[1]
            vertex_pos[5,i] = quat[2]
            vertex_pos[6,i] = quat[3]

        # print(f" vertex {vertex_pos} ")    
        return vertex_pos

    def get_grabbable_tree_links():
        # get branch link indices that can be used for interacting with
        grabbable_link_indices = []
        grabbable_link_poses = []

        idx = 0
        for link_name in tree.link_names:
            if not "base" in link_name and not "tip" in link_name: # Exclude base from being a push option
                grabbable_link_indices.append(idx)
            idx += 1
        # print(f"size of grabbable_link_indices {len(grabbable_link_indices)} ")
        # print(f"grabbable_link_indices {grabbable_link_indices} ")

        grabbable_link_poses = get_link_poses()[:,grabbable_link_indices]
        # print(f"grabbable_link_poses {grabbable_link_poses} ")
        # print(f" size of grabbable_link_poses {grabbable_link_poses.shape} ")

        return grabbable_link_indices, grabbable_link_poses



    policy = GraspTreePolicy(franka, franka_name)

    while True:
        # get grabble tree link poses
        grabbable_link_indices, grabbable_link_poses = get_grabbable_tree_links()


        #randomly choose index to grab
        idx = np.random.randint(0, len(grabbable_link_indices))
        idx = -1 #hardcode to grab the last link

        goal_grab_pose = grabbable_link_poses[:,idx]
        print(f"grabbing link idx {idx}, goal_grab_pose {goal_grab_pose} ")
        policy.set_grasp_goal(goal_grab_pose)

        print(f"resetting policy")
        policy.reset()
        print(f"running policy")
        scene.run(time_horizon=policy.time_horizon, policy=policy)

    