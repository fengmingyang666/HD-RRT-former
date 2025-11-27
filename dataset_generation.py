import pybullet_planning as pp
import pybullet as p
import time
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.transform import Rotation as R
import random
from pybullet_planning.interfaces.robots.collision import pairwise_link_collision
from pybullet_planning.interfaces.robots.link import get_self_link_pairs
import copy
from tqdm import tqdm
import json
import multiprocessing as mp
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

class RobotPlanner:
    def __init__(self, urdf_path="UR3/urdf/UR3.urdf", GUI=True):
        """
        init RobotPlanner
        Args:
            urdf_path: URDF file path
            GUI: whether to enable GUI visualization, default True
        """
        self.urdf_path = urdf_path
        
        self.physics_client = p.connect(p.GUI if GUI else p.DIRECT)
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.joints = pp.get_movable_joints(self.robot)
        self.obstacles = []
        self.obstacles_list = []
        self.init_conf = pp.get_joint_positions(self.robot, self.joints)
        self.tree = []
        self.env_map = None
        
        self.joint_limits = []
        for joint in self.joints:
            lower, upper = p.getJointInfo(self.robot, joint)[8:10]
            if lower > upper:
                lower, upper = -np.pi, np.pi
            self.joint_limits.append((lower, upper))
        
        self.n_control_points = 10
        self.collision_weight = 100.0
        self.smoothness_weight = 1.0
        self.length_weight = 0.1
        self.joint_weight = 0.5
    
    def add_sphere(self, pos, radius, color=(0, 0, 1, 0.5)):
        """
        add a sphere obstacle
        
        Args:
            pos: obstacle position [x, y, z]
            radius: sphere radius
            color: RGBA color, default blue transparent
            
        Returns:
            body ID
        """
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        obstacle_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, 
                                         baseVisualShapeIndex=vis_id, basePosition=pos)
        self.obstacles.append(obstacle_body)
        self.obstacles_list.append((pos, radius))
        return obstacle_body
    
    def visualize_trajectory(self, path):
        """
        visualize the trajectory
        
        Args:
            path: path points list
        """
        if path is None:
            print("Path is empty, cannot visualize")
            return
        
        print("Start visualizing trajectory...")
        print(f"Path length: {len(path)}")

        for i, config in enumerate(path):
            ee_pose = p.getLinkState(self.robot, self.joints[-1])[0]
            pp.set_joint_positions(self.robot, self.joints, config)
            ee_pose_new = p.getLinkState(self.robot, self.joints[-1])[0]
            pp.add_line(ee_pose, ee_pose_new, color=(1, 0, 0, 1), width=2)
            p.stepSimulation()
            time.sleep(20/len(path))
        
        print("Trajectory visualization completed")
    
    def is_collision_free(self, config, return_obstacle=False):
        """
        check if the configuration is collision-free
        
        Args:
            config: joint configuration
            
        Returns:
            True if collision-free, False otherwise
        """
        pp.set_joint_positions(self.robot, self.joints, config)
        collisions = []
        for obstacle in self.obstacles:
            collision = pp.pairwise_collision(self.robot, obstacle) 
            if collision:
                collisions.append(obstacle)
        if return_obstacle:
            return len(collisions) == 0, collisions
        return len(collisions) == 0
    
    def get_robot_links(self):
        """
        get all link IDs of the robot
        
        Returns:
            link ID list
        """
        links = []
        for i in range(p.getNumJoints(self.robot)):
            links.append(p.getJointInfo(self.robot, i)[0])
        return links
    
    def distance_to_obstacles(self, config):
        """
        calculate the minimum distance from the configuration to all obstacles
        
        Args:
            config: joint configuration
            
        Returns:
            minimum distance
        """
        pp.set_joint_positions(self.robot, self.joints, config)
        min_distance = float('inf')

        for link_id in range(p.getNumJoints(self.robot)):
            link_state = p.getLinkState(self.robot, link_id, computeForwardKinematics=True)
            link_pos = link_state[0]

            for obstacle_id in self.obstacles:
                obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
                distance = np.linalg.norm(np.array(link_pos) - np.array(obstacle_pos))

                obstacle_radius = p.getVisualShapeData(obstacle_id)[0][3][0]

                actual_distance = distance - obstacle_radius
                min_distance = min(min_distance, actual_distance)
        
        return min_distance
    
    def smooth_path(self, path):
        """
        smooth the path
        
        Args:
            path: original path
            
        Returns:
            smoothed path
        """
        pass

    def plan_path(self, start_conf=None, goal_conf=None):
        """
        plan a path
        
        Args:
            start_conf: start joint configuration
            goal_conf: goal joint configuration
            use_optimized: whether to use optimized algorithm
            
        Returns:
            planned path
        """
        if start_conf is None:
            start_conf = pp.get_joint_positions(self.robot, self.joints)
            
        if goal_conf is None:
            raise ValueError("goal joint configuration cannot be None")
        
        path = pp.plan_joint_motion(self.robot, self.joints, goal_conf, 
                                    obstacles=self.obstacles, algorithm='birrt', max_time=10)

        return path
    
    def plan_path_linear(self, start_conf=None, goal_conf=None, n_steps=10):
        """
        linear interpolation planning path
        
        Args:
            start_conf: start joint configuration
            goal_conf: goal joint configuration
            
        Returns:
            planned linear path
        """
        if start_conf is None:
            start_conf = pp.get_joint_positions(self.robot, self.joints)
            
        if goal_conf is None:
            raise ValueError("goal joint configuration cannot be None")
        
        
        path = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            conf = [start_conf[j] + t * (goal_conf[j] - start_conf[j]) for j in range(len(start_conf))]
            path.append(conf)
        
        # check collision and remove obstacles
        collision_obs = set()
        for conf in path:
            flag, obstacle = self.is_collision_free(conf, return_obstacle=True)
            if not flag:
                for obs in obstacle:
                    collision_obs.add(obs)
        
        # remove collision obstacles
        for obs in collision_obs:

            obs_idx = self.obstacles.index(obs)
            p.removeBody(obs)
            self.obstacles.remove(obs)
            
            self.obstacles_list.pop(obs_idx)
        
        return path
    
    def disconnect(self):
        """
        disconnect from PyBullet physics engine
        """
        p.disconnect(self.physics_client)
        print("disconnected from PyBullet physics engine")

    def reset(self):
        """
        reset robot to initial state, including complete resource cleanup
        """
        pp.set_joint_positions(self.robot, self.joints, self.init_conf)
        
        if hasattr(self, 'collision_detector') and self.collision_detector:
            self.collision_detector.clear_cache()
        
        for obstacle in self.obstacles:
            p.removeBody(obstacle)
        self.obstacles = []
        self.obstacles_list = []
        
        if hasattr(self, 'trajectory'):
            self.trajectory = None
        if hasattr(self, 'path'):
            self.path = None
        
        p.performCollisionDetection()
        p.stepSimulation()
        
        import gc
        gc.collect()
        
        return

    def in_collision(self, q):
        """check whether the robot is in collision with obstacles
        
        Args:
            q: joint configuration
            
        Returns:
            True if in collision, False otherwise
        """
        q = np.asarray(q)
        # set robot to config
        pp.set_joint_positions(self.robot, self.joints, q)
        # check robot vs obstacles
        for obs in self.obstacles:
            contacts = p.getContactPoints(bodyA=self.robot, bodyB=obs)
            if len(contacts) > 0:
                return True
        # check self-collision using pybullet_planning helper
        # iterate over link pairs
        try:
            pairs = get_self_link_pairs(self.robot)
            for (la, lb) in pairs:
                if pairwise_link_collision(self.robot, la, self.robot, lb):
                    return True
        except Exception:
            # fallback: rely on contact points to detect obvious self-collisions
            contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.robot)
            if len(contacts) > 0:
                return True
        return False

    def self_collision_check(self, conf):
        """
        check self-collision
        
        Args:
            conf: joint configuration
            
        Returns:
            True if in collision, False otherwise
        """
        self_check_link_pairs = get_self_link_pairs(self.robot, self.joints)
        for link1, link2 in self_check_link_pairs:
            if pairwise_link_collision(self.robot, link1, self.robot, link2):
                return True
        return False

    def path_distance(self, q1, q2):
        return np.linalg.norm(np.asarray(q1) - np.asarray(q2))


    # About generating dataset
    def generate_random_configuration(self,max_attempts=1000,min_distance=0.2):
        """
        generate random joint configuration
        
        Returns:
            random joint configuration
        """
        conf = [random.uniform(lower, upper) for lower, upper in self.joint_limits]
        # check self collision
        attempts = 0
        while (self.in_collision(conf) or self.path_distance(self.init_conf, conf) < min_distance) and attempts < max_attempts:
            conf = [random.uniform(lower, upper) for lower, upper in self.joint_limits]
            attempts += 1
        if attempts >= max_attempts:
            print(f"Cannot generate valid configuration after {max_attempts} attempts")
        return conf
    def generate_random_obstacle(self, num_obstacles=1, workspace_radius=0.5, min_radius=0.02, max_radius=0.1):
        """
        generate random obstacle
        
        Returns:
            random obstacle position and radius
        """
        obstacles = []
        for _ in range(num_obstacles):
            position = [random.uniform(-workspace_radius, workspace_radius) for _ in range(3)]
            radius = random.uniform(min_radius, max_radius)
            obstacles.append((position, radius))
            # TODO: check reachability
        return obstacles

    def generate_dataset_config(self, num_samples=1000, num_obstacles=[1,1], workspace_radius=0.5, min_radius=0.02, max_radius=0.1):
        """
        generate dataset config
        
        Returns:
            dataset config, containing random configuration and obstacles
        """
        dataset_cfg = []
        for _ in tqdm(range(num_samples), desc="generate dataset config..."):
            
            num_obstacle = random.randint(num_obstacles[0], num_obstacles[1])
            obs = self.generate_random_obstacle(num_obstacles=num_obstacle, workspace_radius=workspace_radius, min_radius=min_radius, max_radius=max_radius)
            conf = self.generate_random_configuration()
            dataset_cfg.append((conf, obs))
        return dataset_cfg

    def generate_dataset_path(self, dataset_cfg):
        """
        generate dataset path
        
        Returns:
            dataset path, containing random path and obstacles
        """
        dataset_path = []
        num_samples = len(dataset_cfg)
        # plan path
        for i in tqdm(range(num_samples), desc="generate dataset path..."):
            self.reset()
            conf, obs = dataset_cfg[i]
            for pos, radius in obs:
                self.add_sphere(pos, radius)
            n_steps = random.randint(4, 10)
            path = self.plan_path(goal_conf=conf,use_optimized=False)
            # path = self.plan_path_linear(goal_conf=conf)
            if path is not None and len(path) > 32:
                path = random.sample(path, random.randint(4, 32))

            dataset_cfg[i] = (conf, self.obstacles_list)  # update obstacles list
            dataset_path.append(path)
        return dataset_path

    def create_env_map(self, workspace_radius=0.5, voxel_size=0.02):
        '''create environment map'''
        env_map = np.zeros([int(workspace_radius*2/voxel_size), int(workspace_radius*2/voxel_size), int(workspace_radius*2/voxel_size)], dtype=np.uint8)

        for pos, radius in self.obstacles_list:
            for i in range(int((pos[0] - radius)/voxel_size), int((pos[0] + radius)/voxel_size)):
                for j in range(int((pos[1] - radius)/voxel_size), int((pos[1] + radius)/voxel_size)):
                    for k in range(int((pos[2] - radius)/voxel_size), int((pos[2] + radius)/voxel_size)):
                        if 0 <= i < env_map.shape[0] and 0 <= j < env_map.shape[1] and 0 <= k < env_map.shape[2]:
                            if np.linalg.norm((pos[0] - i*voxel_size, pos[1] - j*voxel_size, pos[2] - k*voxel_size)) <= radius:
                                env_map[i, j, k] = 1

        return env_map

    def generate_dataset_env_map(self, dataset_cfg, workspace_radius=0.5, voxel_size=0.02):
        '''generate dataset env map'''
        dataset_env_map = []
        for conf, obs in tqdm(dataset_cfg, desc="generate dataset env map..."):
            self.reset()
            for pos, radius in obs:
                self.add_sphere(pos, radius)
            env_map = self.create_env_map(workspace_radius=workspace_radius, voxel_size=voxel_size)
            dataset_env_map.append(env_map)
        return dataset_env_map
        
    def generate_dataset(self, num_samples=1000, num_obstacles=[1,1], workspace_radius=0.5, voxel_size=0.02, min_radius=0.02, max_radius=0.1, D='6D'):
        '''generate dataset'''
        dataset = []
        dataset_cfg = self.generate_dataset_config(num_samples=num_samples, num_obstacles=num_obstacles, workspace_radius=workspace_radius, min_radius=min_radius, max_radius=max_radius)
        dataset_path = self.generate_dataset_path(dataset_cfg) # may remove obstacles, so must before generate env map
        dataset_env_map = self.generate_dataset_env_map(dataset_cfg, workspace_radius=workspace_radius, voxel_size=voxel_size)
        # dataset_env_map = self.generate_dataset_env_map_mp(dataset_cfg, workspace_radius=workspace_radius, voxel_size=voxel_size)

        if len(dataset_path) != num_samples or len(dataset_env_map) != num_samples:
            raise ValueError("dataset path number and dataset config number are not equal or dataset env map number and dataset config number are not equal")
        # save dataset_cfg and path
        os.makedirs("data", exist_ok=True)
        with open(f"data/dataset_cfg_{D}.json", "w") as f:
            json.dump(dataset_cfg, f, indent=4)
        with open(f"data/dataset_path_{D}.json", "w") as f:
            json.dump(dataset_path, f, indent=4)
        # number of valid samples
        num_valid_samples = 0
        for i in range(num_samples):
            if dataset_path[i] is None:
                continue
            num_valid_samples += 1
        print(f"valid sample number: {num_valid_samples}")
        
        for i in range(num_samples):
            if dataset_path[i] is None:
                continue
            for j in range(1,len(dataset_path[i])):
                dataset.append((dataset_path[i][:j], dataset_path[i][j], dataset_env_map[i]))
        return np.array(dataset, dtype=object)
    
    @classmethod
    def process_single_config(cls, args):
        """process single config"""
        conf_obs, workspace_radius, voxel_size = args
        conf, obs = conf_obs
        
        env_instance = cls(GUI=False)
        env_instance.reset()
        for pos, radius in obs:
            env_instance.add_sphere(pos, radius)
        env_map = env_instance.create_env_map(workspace_radius=workspace_radius, voxel_size=voxel_size)
        return env_map

    def generate_dataset_env_map_mp(self, dataset_cfg, workspace_radius=0.5, voxel_size=0.02):
        '''generate dataset env map - multiprocessing version'''
        tasks = [(conf_obs, workspace_radius, voxel_size) for conf_obs in dataset_cfg]
        
        with mp.Pool(processes=2) as pool:
            results = list(tqdm(pool.imap(self.__class__.process_single_config, tasks), 
                            total=len(tasks), 
                            desc="generate dataset env map..."))
        
        return results

def single_test():
    """
    single test example
    """
    planner = RobotPlanner()
    
    goal_conf = [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]

    planner.add_sphere([0.25, 0, 0.25], 0.05)
    planner.add_sphere([0.35, 0, 0.05], 0.05)
    
    path = planner.plan_path(goal_conf=goal_conf)
    # path = planner.plan_path_linear(goal_conf=goal_conf, n_steps=10)
    print(f"plan path: {path}")
    
    planner.visualize_trajectory(path)
    

def gen_data_test(D='6D'):
    os.makedirs("data", exist_ok=True)
    if D == '6D':
        urdf_path = "UR3/urdf/UR3.urdf"
    elif D == '7D':
        urdf_path = "franka_description/robots/panda_arm.urdf"
    planner = RobotPlanner(urdf_path=urdf_path, GUI=False)

    dataset = planner.generate_dataset(num_samples=8000, num_obstacles=[10,20], workspace_radius=0.5, voxel_size=0.02, min_radius=0.02, max_radius=0.1,D=D)
    print(f"generate dataset number: {len(dataset)}")
    np.save(f"data/dataset_{D}.npy", dataset)

if __name__ == "__main__":
    set_seed(42)
    # single_test()
    gen_data_test(D='6D')
    