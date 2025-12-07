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
import logging
import torch
import time
import utils

logging.basicConfig(level=logging.INFO)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

class RobotPlanner:
    def __init__(self, urdf_path="UR3/urdf/UR3.urdf", GUI=True):
        """
        init robot planner
        
        Args:
            urdf_path: robot urdf path
            GUI: whether to enable GUI visualization, default True
        """
        self.urdf_path=urdf_path
        self.physics_client = p.connect(p.GUI if GUI else p.DIRECT)
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.joints = pp.get_movable_joints(self.robot)

        self.obstacles = []
        self.obstacles_list = []
        self.init_conf = pp.get_joint_positions(self.robot, self.joints)
        self.tree = []
        self.env_map = None
        self.ghost = []
        
        self.joint_limits = []
        for joint in self.joints:
            lower, upper = p.getJointInfo(self.robot, joint)[8:10]

            if lower > upper:
                lower, upper = -np.pi, np.pi
            self.joint_limits.append((lower, upper))
        self.joint_limits = np.array(self.joint_limits)


    def create_ghost_robot_advanced(self, alpha=0.3, color=[1, 0.5, 0]):
        """
        generate transparent ghost robot at current pose
        - using loadURDF, not participate in collision
        - modify link visual shape color to transparent
        - sync joint angle
        """

        ghost = p.loadURDF(self.urdf_path, useFixedBase=True)

        num_joints = p.getNumJoints(ghost)
        for link in range(-1, num_joints):
            p.setCollisionFilterGroupMask(ghost, link, 0, 0)

        visual_data = p.getVisualShapeData(ghost)
        for v in visual_data:
            body_uid, link_idx, geom, dims, filename, lpos, lorn, rgba = v
            rgba_new = list(color) + [alpha]

            p.changeVisualShape(
                objectUniqueId=ghost,
                linkIndex=link_idx,
                rgbaColor=rgba_new
            )

        current_q = pp.get_joint_positions(self.robot, self.joints)
        for j, q in zip(self.joints, current_q):
            p.resetJointState(ghost, j, q)

        return ghost





    def reset(self):
        """
        - set joint position to init_conf
        - clear tree
        - clear env_map
        - clear collision detector cache
        """
        pp.set_joint_positions(self.robot, self.joints, self.init_conf)
        self.tree = []
        self.env_map = None

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
    
    def create_env_map(self, workspace_radius=0.5, voxel_size=0.02):
        '''create voxel map of the environment'''
        env_map = np.zeros([int(workspace_radius*2/voxel_size), int(workspace_radius*2/voxel_size), int(workspace_radius*2/voxel_size)], dtype=np.uint8)

        for pos, radius in self.obstacles_list:
            for i in range(int((pos[0] - radius)/voxel_size), int((pos[0] + radius)/voxel_size)):
                for j in range(int((pos[1] - radius)/voxel_size), int((pos[1] + radius)/voxel_size)):
                    for k in range(int((pos[2] - radius)/voxel_size), int((pos[2] + radius)/voxel_size)):
                        if 0 <= i < env_map.shape[0] and 0 <= j < env_map.shape[1] and 0 <= k < env_map.shape[2]:
                            if np.linalg.norm((pos[0] - i*voxel_size, pos[1] - j*voxel_size, pos[2] - k*voxel_size)) <= radius:
                                env_map[i, j, k] = 1

        return env_map


    def set_joint_positions(self, q):
        """set robot joint positions (directly set movable joints)"""
        for j, v in zip(self.joints, q):
            p.resetJointState(self.robot, j, v)
        # force collision detection update
        p.performCollisionDetection()

    def within_limits(self, q):
        q = np.asarray(q)
        lows = self.joint_limits[:,0]
        highs = self.joint_limits[:,1]
        return np.all(q >= lows - 1e-8) and np.all(q <= highs + 1e-8)

    def in_collision(self, q):
        """check if given joint configuration is in collision (including self-collision)
        return True if in collision
        """
        q = np.asarray(q)
        if not self.within_limits(q):
            return True
        # set robot to config
        self.set_joint_positions(q)
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
        check self-collision for given joint configuration
        
        Args:
            conf: joint configuration
            
        Returns:
            True if in collision
        """
        self_check_link_pairs = get_self_link_pairs(self.robot, self.joints)
        for link1, link2 in self_check_link_pairs:
            if pairwise_link_collision(self.robot, link1, self.robot, link2):
                return True
        return False

    def sample_random_config(self):
        """sample a random configuration in joint space (uniformly)"""
        lows = self.joint_limits[:,0]
        highs = self.joint_limits[:,1]
        node_next = np.random.uniform(lows, highs)
        # print("random node next:",node_next)
        return node_next

    def sample_transformer_config(self,model,device):
        """
        sample a configuration using Transformer model
        """
        # print("sample transformer config")
        tree_nodes = list()
        if len(self.tree) == 0:
            return self.sample_random_config()
        for node in self.tree:
            tree_nodes.append(node['q'])
        # print(len(tree_nodes))

        tree_nodes_tensor = torch.tensor(np.array(tree_nodes), dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 6)
        env_map_tensor = torch.tensor(self.env_map, dtype=torch.float32).unsqueeze(0).to(device)  # (1, depth, height, width)

        with torch.no_grad():
            next_sample_point = model(tree_nodes_tensor, env_map_tensor)

        node_next = next_sample_point.cpu().numpy()[0]
        # print("transformer node next:",node_next)
        return node_next

    def nearest_node(self, q_rand):
        """find index of nearest node in tree (using Euclidean distance)"""
        dists = [np.linalg.norm(np.asarray(node['q']) - q_rand) for node in self.tree]
        return int(np.argmin(dists))

    def steer(self, q_near, q_rand, step_size):
        """from q_near towards q_rand move step_size in joint space"""
        q_near = np.asarray(q_near)
        q_rand = np.asarray(q_rand)
        vec = q_rand - q_near
        dist = np.linalg.norm(vec)
        if dist <= step_size:
            q_new = q_rand
            # q_new = q_near
        else:
            q_new = q_near + vec / dist * step_size
        # clamp to joint limits
        lows = self.joint_limits[:,0]
        highs = self.joint_limits[:,1]
        q_new = np.minimum(np.maximum(q_new, lows), highs)
        return q_new

    def path_distance(self, q1, q2):
        return np.linalg.norm(np.asarray(q1) - np.asarray(q2))

    def interpolate_check(self, q1, q2, step_size):
        """linearly interpolate between q1 and q2 with step_size and check collision (including endpoints)"""
        q1 = np.asarray(q1)
        q2 = np.asarray(q2)
        dist = np.linalg.norm(q2 - q1)
        if dist == 0:
            return not self.in_collision(q1)
        n_steps = int(np.ceil(dist / (step_size * 0.5)))
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q = (1 - alpha) * q1 + alpha * q2
            if self.in_collision(q):
                return False
        return True

    def plan_rrt(self, start=None, goal=None, max_iters=5000, step_size=0.2,
                 goal_thresh=0.15, goal_bias=0.05, verbose=True):
        """
        RRT algorithm in joint space

        Args:
            start: start joint configuration (None to use current robot state)
            goal: goal joint configuration (must provide)
            max_iters: maximum number of iterations
            step_size: step size in joint space
            goal_thresh: threshold to reach goal
            goal_bias: probability to sample goal

        Returns:
            path: joint configuration list from start to goal (None if failed)
        """
        if goal is None:
            raise ValueError("goal must be provided")
        if start is None:
            start = pp.get_joint_positions(self.robot, self.joints)
        start = np.asarray(start)
        goal = np.asarray(goal)
        if self.in_collision(start):
            logging.warning("Start configuration is in collision, skipping")
            return None, True
        if self.in_collision(goal):
            logging.warning("Goal configuration is in collision, skipping")
            return None, True

        self.tree = []
        self.env_map = self.create_env_map()
        self.tree.append({'q': start.copy(), 'parent': -1})

        for it in range(max_iters):
            # sample
            if random.random() < goal_bias:
                q_rand = goal.copy()
            else:
                q_rand = self.sample_random_config()
            # find nearest
            idx_near = self.nearest_node(q_rand)
            q_near = np.asarray(self.tree[idx_near]['q'])
            # steer
            q_new = self.steer(q_near, q_rand, step_size)
            # collision checking along the segment
            if not self.interpolate_check(q_near, q_new, step_size):
                continue
            # add node
            self.tree.append({'q': q_new.copy(), 'parent': idx_near})
            # check goal
            if self.path_distance(q_new, goal) <= goal_thresh:
                # try connect directly to goal
                if self.interpolate_check(q_new, goal, step_size):
                    self.tree.append({'q': goal.copy(), 'parent': len(self.tree) - 1})
                    if verbose:
                        print(f"RRT: found path at iter {it}")
                    return self.reconstruct_path(len(self.tree) - 1), len(self.tree), it
        if verbose:
            print("RRT: failed to find path")
        return None, False

    def plan_rrt_star(self, start=None, goal=None,alpha=0.0, max_iters=5000, step_size=0.2,
                      goal_thresh=0.15, goal_bias=0.05, gamma=1.5, verbose=True,model=None,device=None):
        """
        RRT* in joint space.
        Returns (path, total_nodes, iters) on success, else None.
        """
        if goal is None:
            raise ValueError("goal must be provided")
        if start is None:
            start = pp.get_joint_positions(self.robot, self.joints)
        start = np.asarray(start)
        goal = np.asarray(goal)

        if self.in_collision(start):
            logging.warning("Start configuration is in collision, skipping")
            return None, True
        if self.in_collision(goal):
            logging.warning("Goal configuration is in collision, skipping")
            return None, True

        # init
        self.tree = []
        self.env_map = self.create_env_map()
        # node: {'q': q, 'parent': parent_idx, 'cost': cost}
        self.tree.append({'q': start.copy(), 'parent': -1, 'cost': 0.0})

        dim = len(self.joints)

        def nearest_index(q):
            dists = [np.linalg.norm(np.asarray(n['q']) - q) for n in self.tree]
            return int(np.argmin(dists))

        def near_indices(q, n_nodes):
            # radius as in RRT*: gamma * (log(n)/n)^(1/d)
            if n_nodes <= 1:
                return []
            radius = gamma * (np.log(n_nodes) / n_nodes) ** (1.0 / dim)
            # scale radius by joint-space span (roughly) to avoid extremely small numbers
            # but keep consistent units (rad): we won't rescale here; user can tune gamma
            inds = [i for i, node in enumerate(self.tree) if np.linalg.norm(np.asarray(node['q']) - q) <= radius]
            return inds

        for it in range(max_iters):
            # sample
            if random.random() < goal_bias:
                q_rand = goal.copy()
            elif random.random() < alpha:
                q_rand = self.sample_transformer_config(model,device)
            else:
                q_rand = self.sample_random_config()
            # nearest
            idx_near = nearest_index(q_rand)
            q_near = np.asarray(self.tree[idx_near]['q'])
            # steer
            q_new = self.steer(q_near, q_rand, step_size)
            # collision along
            if not self.interpolate_check(q_near, q_new, step_size):
                continue
            # compute cost to come via nearest
            cost_to_new = self.tree[idx_near]['cost'] + np.linalg.norm(q_new - q_near)
            # find near nodes for possible better parent
            near_inds = near_indices(q_new, len(self.tree))
            best_parent = idx_near
            best_cost = cost_to_new
            for ni in near_inds:
                q_neighbor = np.asarray(self.tree[ni]['q'])
                # check if path from neighbor to q_new is collision free
                if not self.interpolate_check(q_neighbor, q_new, step_size):
                    continue
                cost_via_neighbor = self.tree[ni]['cost'] + np.linalg.norm(q_new - q_neighbor)
                if cost_via_neighbor < best_cost:
                    best_cost = cost_via_neighbor
                    best_parent = ni
            # add node
            self.tree.append({'q': q_new.copy(), 'parent': best_parent, 'cost': best_cost})
            new_idx = len(self.tree) - 1
            # rewire: try to see if going from new node to neighbors reduces their cost
            for ni in near_inds:
                if ni == best_parent:
                    continue
                q_neighbor = np.asarray(self.tree[ni]['q'])
                if not self.interpolate_check(q_new, q_neighbor, step_size):
                    continue
                cost_through_new = self.tree[new_idx]['cost'] + np.linalg.norm(q_neighbor - q_new)
                if cost_through_new + 1e-9 < self.tree[ni]['cost']:
                    # rewire
                    self.tree[ni]['parent'] = new_idx
                    self.tree[ni]['cost'] = cost_through_new

            # check goal reachability
            if self.path_distance(q_new, goal) <= goal_thresh:
                if self.interpolate_check(q_new, goal, step_size):
                    # append goal as final node
                    goal_parent_idx = new_idx
                    goal_cost = self.tree[new_idx]['cost'] + np.linalg.norm(goal - q_new)
                    self.tree.append({'q': goal.copy(), 'parent': goal_parent_idx, 'cost': goal_cost})
                    if verbose:
                        print(f"RRT*: found path at iter {it}")
                    return self.reconstruct_path(len(self.tree) - 1), len(self.tree), it
        if verbose:
            print("RRT*: failed to find path")
        return None, False

    def plan_rrt_connect(self, start=None, goal=None,alpha=0, max_iters=5000, step_size=0.2,
                         goal_thresh=0.15, goal_bias=0.05, max_connect_iters=50,model=None, device=None, verbose=True):
        """
        RRT-Connect (bidirectional RRT). Grows two trees and tries to connect them.
        Returns (path, total_nodes, iters) on success, else None.
        """
        if goal is None:
            raise ValueError("goal must be provided")
        if start is None:
            start = pp.get_joint_positions(self.robot, self.joints)
        start = np.asarray(start)
        goal = np.asarray(goal)

        if self.in_collision(start):
            logging.warning("Start configuration is in collision, skipping")
            return None, True
        if self.in_collision(goal):
            logging.warning("Goal configuration is in collision, skipping")
            return None, True

        # Trees: list of nodes dict {'q': q, 'parent': parent_idx}
        self.env_map = self.create_env_map()
        tree_a = [{'q': start.copy(), 'parent': -1}]
        tree_b = [{'q': goal.copy(), 'parent': -1}]
        

        def nearest_index_tree(tree, q):
            dists = [np.linalg.norm(np.asarray(n['q']) - q) for n in tree]
            return int(np.argmin(dists))

        def extend(tree, q_target):
            """
            Try to extend tree towards q_target by one step (steer).
            Returns (status, new_idx)
              status: "advanced" if new node added, "reached" if reached target, "trapped" if blocked
            """
            idx_near = nearest_index_tree(tree, q_target)
            q_near = np.asarray(tree[idx_near]['q'])
            q_new = self.steer(q_near, q_target, step_size)
            if not self.interpolate_check(q_near, q_new, step_size):
                return "trapped", None
            # add node
            tree.append({'q': q_new.copy(), 'parent': idx_near})
            new_idx = len(tree) - 1
            if np.linalg.norm(q_new - q_target) <= 1e-8:
                return "reached", new_idx
            if self.path_distance(q_new, q_target) <= goal_thresh and self.interpolate_check(q_new, q_target, step_size):
                # can directly reach
                tree.append({'q': q_target.copy(), 'parent': new_idx})
                return "reached", len(tree) - 1
            return "advanced", new_idx

        def connect(tree, q_target):
            """
            Repeatedly extend tree towards q_target until cannot advance or reached.
            Returns (status, last_idx)
            """
            last_status = None
            last_idx = None
            for _ in range(max_connect_iters):
                status, idx = extend(tree, q_target)
                last_status = status
                last_idx = idx
                if status != "advanced":
                    break
            return last_status, last_idx

        # Main loop: alternate growth
        for it in range(max_iters):
            self.tree = tree_a + tree_b
            # sample random config biased toward goal with goal_bias
            if random.random() < goal_bias:
                q_rand = goal.copy()
            elif random.random() < alpha:
                q_rand = self.sample_transformer_config(model,device)
            else:
                q_rand = self.sample_random_config()

            # extend tree A towards q_rand
            status_a, idx_a = extend(tree_a, q_rand)
            if status_a in ("advanced", "reached"):
                # try to connect tree B to the new node in A
                q_new_a = tree_a[idx_a]['q'] if idx_a is not None else tree_a[-1]['q']
                status_b, idx_b = connect(tree_b, q_new_a)
                if status_b == "reached" or (status_b == "advanced" and np.linalg.norm(tree_b[-1]['q'] - q_new_a) <= goal_thresh):
                    # found connection between tree_a and tree_b -> build path
                    # find connect indices: last nodes in both trees that are close
                    # we look for a pair of nodes (ia, ib) such that distance small
                    connect_pair = None
                    for ia, na in enumerate(tree_a):
                        for ib, nb in enumerate(tree_b):
                            if np.linalg.norm(np.asarray(na['q']) - np.asarray(nb['q'])) <= goal_thresh:
                                connect_pair = (ia, ib)
                                break
                        if connect_pair:
                            break
                    if connect_pair is None:
                        # fallback: use last indices
                        ia = len(tree_a) - 1
                        ib = len(tree_b) - 1
                    else:
                        ia, ib = connect_pair

                    # reconstruct path from start to ia
                    def build_path_from_tree(tree, idx):
                        path = []
                        cur = idx
                        while cur != -1:
                            path.append(np.asarray(tree[cur]['q']).copy())
                            cur = tree[cur]['parent']
                        path.reverse()
                        return path

                    path_a = build_path_from_tree(tree_a, ia)  # start -> ia
                    path_b = build_path_from_tree(tree_b, ib)  # goal -> ib
                    # path_b is from goal->...->ib, need to reverse (ib->goal) and concatenate
                    path_b_rev = [np.asarray(q).copy() for q in path_b]
                    path_b_rev.reverse()
                    # connect ensuring continuity: last of path_a ~ first of path_b_rev
                    full_path = path_a + path_b_rev
                    if verbose:
                        print(f"RRT-Connect: found path at iter {it}")
                    # merge trees into self.tree for compatibility (simple append)
                    self.tree = []
                    for n in tree_a:
                        self.tree.append({'q': n['q'], 'parent': n['parent']})
                    # offset for B
                    offset = len(self.tree)
                    for n in tree_b:
                        self.tree.append({'q': n['q'], 'parent': n['parent'] if n['parent'] == -1 else n['parent'] + offset})
                    return [np.asarray(q).tolist() for q in full_path], len(self.tree), it
            # swap roles A<->B for next iteration
            tree_a, tree_b = tree_b, tree_a

        if verbose:
            print("RRT-Connect: failed to find path")
        return None, False

    def reconstruct_path(self, goal_idx):
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(np.asarray(self.tree[idx]['q']).copy())
            idx = self.tree[idx]['parent']
        path.reverse()
        return path

    def visualize_trajectory(self, path, time_viz = 10, ghosts=False):
        """
        visualize the trajectory of the robot in pybullet
        
        Args:
            path: path points list
        """
        if path is None:
            print("path is None, cannot visualize")
            return
        
        print("visualize the trajectory of the robot in pybullet...")
        print(f"path length: {len(path)}")

        for i, config in enumerate(path):
            ee_pose = p.getLinkState(self.robot, self.joints[-1])[0]
            pp.set_joint_positions(self.robot, self.joints, config)
            ee_pose_new = p.getLinkState(self.robot, self.joints[-1])[0]
            if i > 0:
                pp.add_line(ee_pose, ee_pose_new, color=(1, 0, 0, 1), width=2)
            p.stepSimulation()
            time.sleep(time_viz/len(path))
            # ghost robot
            if ghosts:
                t = i/max(1,len(path))
                start_color = [0,1,0] # green
                end_color=[0,0,1] # blue
                color = [
                    start_color[c] * (1 - t) + end_color[c] * t
                    for c in range(3)
                ]
                alpha=0.3

                if i%6==0:
                    ghost_id = self.create_ghost_robot_advanced(alpha=alpha,color=color)
            
        print("visualize trajectory in pybullet done")
        input("continue...")
        
    def generate_random_configuration(self,max_attempts=1000,min_distance=3):
        """
        generate random joint configuration
        
        Returns:
            random joint configuration
        """
        conf = [random.uniform(lower, upper) for lower, upper in self.joint_limits]
        # check self collision
        attempts = 0
        while (self.in_collision(conf) or self.path_distance(self.init_conf, conf) < min_distance) and attempts < max_attempts:
            # print(self.path_distance(self.init_conf, conf))
            conf = [random.uniform(lower, upper) for lower, upper in self.joint_limits]
            attempts += 1
        if attempts >= max_attempts:
            logging.warning(f"Cannot generate valid configuration after {max_attempts} attempts")
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
            if self.in_collision(self.init_conf):
                continue
            obstacles.append((position, radius))
            
        return obstacles

    def get_path_length(self, path):
        """
        calculate path length
        
        Args:
            path: path points list
        
        Returns:
            path length
        """
        if path is None or len(path) < 2:
            return 0.0
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(np.asarray(path[i]) - np.asarray(path[i+1]))
        return total_length

    def plan(self, method,start=None, goal=None, alpha=0, max_iters=2000, step_size=0.2, verbose=True,model=None,device=None):
        """
        plan path
        
        Args:
            method: planning method ('rrt', 'rrt_connect', 'rrt_star')
            alpha: blending parameter (0-1)
            max_iters: maximum number of iterations
            step_size: step size
            verbose: whether to print verbose information
        
        Returns:
            path points list (if found)
        """
        if method == 'rrt':
            return self.plan_rrt(start=start, goal=goal, max_iters=max_iters, step_size=step_size, verbose=verbose)
        elif method == 'rrt_connect':
            return self.plan_rrt_connect(start=start, goal=goal, alpha=alpha, max_iters=max_iters, step_size=step_size, verbose=verbose,model=model,device=device)
        elif method == 'rrt_star':
            return self.plan_rrt_star(start=start, goal=goal,alpha=alpha, max_iters=max_iters, step_size=step_size, verbose=verbose,model=model, device=device)
        else:
            raise ValueError(f"Unknown planning method: {method}")

def single_test(D='6D',model='ckpt/model_6D.pt'):
    set_seed(42)
    if D == '6D':
        planner = RobotPlanner(urdf_path="UR3/urdf/UR3.urdf", GUI=True)
    elif D == '7D':
        planner = RobotPlanner(urdf_path="franka_description/robots/panda_arm.urdf", GUI=True)
    if D == '6D':
        start = pp.get_joint_positions(planner.robot, planner.joints)
    elif D == '7D':
        start = [-0.6608420449984611, 1.7409806709717686, 2.7333259967081145, -1.2600147118544338, 0.6968546770353532, 1.122730039048114, -2.688324265702779]
        planner.init_conf = start

    obs = planner.generate_random_obstacle(num_obstacles=10)
    # print(obs, start)
    for pos, radius in obs:
        planner.add_sphere(pos, radius)
    goal = planner.generate_random_configuration()
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model,map_location=device,weights_only=False)
    model.eval()

    res = planner.plan_rrt_connect(start=start,goal=goal,alpha=1.0,model=model,device=device)
    # res = planner.plan_rrt(start=start, goal=goal, max_iters=2000, step_size=0.2, verbose=True)
    if res[0] is not None:
        path, expanded_nodes, iters = res
        print(f"Found path with {len(path)} waypoints")

        print(utils.compute_acceleration_value(np.array(path)))
        print(utils.compute_curvature_value(np.array(path)))
        print(utils.compute_energy_value(np.array(path)))
        print(utils.compute_jerk_value(np.array(path)))
        planner.visualize_trajectory(path,ghosts=True)
    else:
        print("No path found")

def test_planner(seed=42,method='rrt_connect',model_path=None, episode=100, alpha=1.0, max_iters=2000, step_size=0.2,D='6D'):
    set_seed(seed)
    if D == '6D':
        planner = RobotPlanner(urdf_path="UR3/urdf/UR3.urdf", GUI=False)
    elif D == '7D':
        planner = RobotPlanner(urdf_path="franka_description/robots/panda_arm.urdf", GUI=False)
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path,map_location=device,weights_only=False)
    model.eval()
    
    
    # metrics
    path_length = []
    expanded_nodes = []
    path_nodes = []
    iters = []
    times = []
    accelerate = []
    jerk = []
    energy_func = []
    curvature = []
    valid_episode = 0
    pbar = tqdm(total=episode)
        
    while valid_episode < episode:
        planner.reset()
        if D == '6D':
            start = pp.get_joint_positions(planner.robot, planner.joints)
        elif D == '7D':
            start = [-0.6608420449984611, 1.7409806709717686, 2.7333259967081145, -1.2600147118544338, 0.6968546770353532, 1.122730039048114, -2.688324265702779]
            planner.init_conf = start
        
        obs = planner.generate_random_obstacle(num_obstacles=random.randint(10, 20))
        for pos, radius in obs:
            planner.add_sphere(pos, radius)
        goal = planner.generate_random_configuration()   
        start_time = time.time()
        # res = planner.plan_net(start=start, goal=goal, alpha=alpha, max_iters=max_iters, step_size=step_size, verbose=True, model=model, device=device)
        res = planner.plan(method=method, start=start, goal=goal, alpha=alpha, max_iters=max_iters, step_size=step_size, verbose=True, model=model, device=device)
        end_time = time.time()
        if res[0] is not None:
            path, expanded_node, iter = res
            print(f"Found path with {len(path)} waypoints")
            # planner.visualize_trajectory(path)
            # metrics
            pbar.update(1)
            valid_episode += 1
            path_length.append(planner.get_path_length(path))
            expanded_nodes.append(expanded_node)
            path_nodes.append(len(path))
            iters.append(iter)
            times.append(end_time - start_time)
            accelerate.append(utils.compute_acceleration_value(np.array(path)))
            jerk.append(utils.compute_jerk_value(np.array(path)))
            energy_func.append(utils.compute_energy_value(np.array(path)))
            curvature.append(utils.compute_curvature_value(np.array(path)))
        else:
            print("No path")

    
    logging.info(f"{method}")
    logging.info(f"alpha: {alpha}")
    logging.info(f'model: {model_path}')
    logging.info(f"Valid episode: {valid_episode}")

    logging.info(f"Average path length: {np.mean(path_length):.2f} ± {np.std(path_length):.2f}")
    logging.info(f"Average expanded nodes: {np.mean(expanded_nodes):.2f} ± {np.std(expanded_nodes):.2f}")
    logging.info(f"Average path nodes: {np.mean(path_nodes):.2f} ± {np.std(path_nodes):.2f}")
    logging.info(f"Average iters: {np.mean(iters):.2f} ± {np.std(iters):.2f}")
    logging.info(f"Average time: {np.mean(times):.2f} ± {np.std(times):.2f}")
    logging.info(f"Average acceleration: {np.mean(accelerate):.2f} ± {np.std(accelerate):.2f}")
    logging.info(f"Average jerk: {np.mean(jerk):.2f} ± {np.std(jerk):.2f}")
    logging.info(f"Average energy: {np.mean(energy_func):.2f} ± {np.std(energy_func):.2f}")
    logging.info(f"Average curvature: {np.mean(curvature):.2f} ± {np.std(curvature):.2f}")
    
    return {'valid_episode':valid_episode, 'path_length':path_length, 'expanded_nodes':expanded_nodes, 'path_nodes':path_nodes, 'iters':iters, 'times':times, 'accelerate':accelerate, 'jerk':jerk, 'energy_func':energy_func, 'curvature':curvature}

import click
import pathlib

@click.command()
@click.option('--seed', default=2025, help='Random seed')
@click.option('--method', default='rrt_connect', help='Planner method. Choose from [rrt, rrt_connect, rrt_star].')
@click.option('--episode', default=100, help='Number of episodes')
@click.option('--alpha', default=1.0, help='Alpha value for Net planner')
@click.option('--max_iters', default=2000, help='Maximum iterations')
@click.option('--step_size', default=0.2, help='Step size')
@click.option('--model',default='ckpt/model_6D.pt')
@click.option('--D', default='6D', help='Dimension of the space. Choose from [6D, 7D].')
def main(seed, method, episode, alpha, max_iters, step_size,model,D):
    result = test_planner(seed=seed, method=method,model_path=model, episode=episode, alpha=alpha, max_iters=max_iters, step_size=step_size,D=D)
    with open(f"res/result_{method}_{alpha}_{pathlib.Path(model).stem}.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
