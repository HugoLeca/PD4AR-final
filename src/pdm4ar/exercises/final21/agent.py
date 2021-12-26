from typing import Sequence

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry, SpacecraftParameters
from shapely.affinity import translate
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
import math
import random
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from dg_commons.maps.shapely_viz import ShapelyViz
import os


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do NOT modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def path(self, start, goal, obstacle_list, rand_area):
        class RRT:
            """
            Class for RRT planning
            """

            class Node:
                """
                RRT Node
                """

                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    self.path_x = []
                    self.path_y = []
                    self.parent = None
                    self.cost = 0.0

            class AreaBounds:

                def __init__(self, area):
                    self.xmin = float(area[0])
                    self.xmax = float(area[1])
                    self.ymin = float(area[2])
                    self.ymax = float(area[3])

            def __init__(self,
                         start,
                         goal,
                         obstacle_list,
                         rand_area,
                         expand_dis=15,
                         path_resolution=1,
                         goal_sample_rate=20,
                         max_iter=1000,
                         connect_circle_dist=50.0,
                         search_until_max_iter=True

                         ):
                """
                Setting Parameter

                start:Start Position [x,y]
                goal:Goal Position [x,y]
                obstacleList:obstacle Positions [[x,y,size],...]
                randArea:Random Sampling Area [min,max]
                play_area:stay inside this area [xmin,xmax,ymin,ymax]

                """
                self.start = self.Node(start[0], start[1])
                self.end = self.Node(goal[0], goal[1])
                self.min_rand = rand_area[0]
                self.max_rand = rand_area[1]
                self.expand_dis = expand_dis
                self.path_resolution = path_resolution
                self.goal_sample_rate = goal_sample_rate
                self.max_iter = max_iter
                self.obstacle_list = obstacle_list
                self.node_list = []
                self.connect_circle_dist = connect_circle_dist
                self.goal_node = self.Node(goal[0], goal[1])
                self.search_until_max_iter = search_until_max_iter

            def planning(self):
                """
                rrt star path planning
                animation: flag for animation on or off .
                """

                self.node_list = [self.start]
                for i in range(self.max_iter):
                    #print("Iter:", i, ", number of nodes:", len(self.node_list))
                    rnd = self.get_random_node()
                    nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
                    new_node = self.steer(self.node_list[nearest_ind], rnd,
                                          self.expand_dis)
                    near_node = self.node_list[nearest_ind]
                    new_node.cost = near_node.cost + \
                                    math.hypot(new_node.x - near_node.x,
                                               new_node.y - near_node.y)

                    if self.check_collision(new_node, self.obstacle_list):
                        near_inds = self.find_near_nodes(new_node)
                        node_with_updated_parent = self.choose_parent(
                            new_node, near_inds)
                        if node_with_updated_parent:
                            self.rewire(node_with_updated_parent, near_inds)
                            self.node_list.append(node_with_updated_parent)
                        else:
                            self.node_list.append(new_node)

                    '''if animation:
                        self.draw_graph(rnd)'''

                    if ((not self.search_until_max_iter)
                            and new_node):  # if reaches goal
                        last_index = self.search_best_goal_node()
                        if last_index is not None:
                            return self.generate_final_course(last_index)

                print("reached max iteration")

                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

                return None

            def choose_parent(self, new_node, near_inds):
                """
                Computes the cheapest point to new_node contained in the list
                near_inds and set such a node as the parent of new_node.
                    Arguments:
                    --------
                        new_node, Node
                            randomly generated node with a path from its neared point
                            There are not coalitions between this node and th tree.
                        near_inds: list
                            Indices of indices of the nodes what are near to new_node
                    Returns.
                    ------
                        Node, a copy of new_node
                """
                if not near_inds:
                    return None

                # search nearest cost in near_inds
                costs = []
                for i in near_inds:
                    near_node = self.node_list[i]
                    t_node = self.steer(near_node, new_node)
                    if t_node and self.check_collision(t_node, self.obstacle_list):
                        costs.append(self.calc_new_cost(near_node, new_node))
                    else:
                        costs.append(float("inf"))  # the cost of collision node
                min_cost = min(costs)

                if min_cost == float("inf"):
                    print("There is no good path.(min_cost is inf)")
                    return None

                min_ind = near_inds[costs.index(min_cost)]
                new_node = self.steer(self.node_list[min_ind], new_node)
                new_node.cost = min_cost

                return new_node

            def search_best_goal_node(self):
                dist_to_goal_list = [
                    self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
                ]
                goal_inds = [
                    dist_to_goal_list.index(i) for i in dist_to_goal_list
                    if i <= self.expand_dis
                ]

                safe_goal_inds = []
                for goal_ind in goal_inds:
                    t_node = self.steer(self.node_list[goal_ind], self.goal_node)
                    if self.check_collision(t_node, self.obstacle_list):
                        safe_goal_inds.append(goal_ind)

                if not safe_goal_inds:
                    return None

                min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
                for i in safe_goal_inds:
                    if self.node_list[i].cost == min_cost:
                        return i

                return None

            def find_near_nodes(self, new_node):
                """
                1) defines a ball centered on new_node
                2) Returns all nodes of the three that are inside this ball
                    Arguments:
                    ---------
                        new_node: Node
                            new randomly generated node, without collisions between
                            its nearest node
                    Returns:
                    -------
                        list
                            List with the indices of the nodes inside the ball of
                            radius r
                """
                nnode = len(self.node_list) + 1
                r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
                # if expand_dist exists, search vertices in a range no more than
                # expand_dist
                if hasattr(self, 'expand_dis'):
                    r = min(r, self.expand_dis)
                dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
                             for node in self.node_list]
                near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
                return near_inds

            def rewire(self, new_node, near_inds):
                """
                    For each node in near_inds, this will check if it is cheaper to
                    arrive to them from new_node.
                    In such a case, this will re-assign the parent of the nodes in
                    near_inds to new_node.
                    Parameters:
                    ----------
                        new_node, Node
                            Node randomly added which can be joined to the tree
                        near_inds, list of uints
                            A list of indices of the self.new_node which contains
                            nodes within a circle of a given radius.
                    Remark: parent is designated in choose_parent.
                """
                for i in near_inds:
                    near_node = self.node_list[i]
                    edge_node = self.steer(new_node, near_node)
                    if not edge_node:
                        continue
                    edge_node.cost = self.calc_new_cost(new_node, near_node)

                    no_collision = self.check_collision(edge_node, self.obstacle_list)
                    improved_cost = near_node.cost > edge_node.cost

                    if no_collision and improved_cost:
                        near_node.x = edge_node.x
                        near_node.y = edge_node.y
                        near_node.cost = edge_node.cost
                        near_node.path_x = edge_node.path_x
                        near_node.path_y = edge_node.path_y
                        near_node.parent = edge_node.parent
                        self.propagate_cost_to_leaves(new_node)

            def calc_new_cost(self, from_node, to_node):
                d, _ = self.calc_distance_and_angle(from_node, to_node)
                return from_node.cost + d

            def propagate_cost_to_leaves(self, parent_node):

                for node in self.node_list:
                    if node.parent == parent_node:
                        node.cost = self.calc_new_cost(parent_node, node)
                        self.propagate_cost_to_leaves(node)

            def steer(self, from_node, to_node, extend_length=float("inf")):

                new_node = self.Node(from_node.x, from_node.y)
                d, theta = self.calc_distance_and_angle(new_node, to_node)

                new_node.path_x = [new_node.x]
                new_node.path_y = [new_node.y]

                if extend_length > d:
                    extend_length = d

                n_expand = math.floor(extend_length / self.path_resolution)

                for _ in range(n_expand):
                    new_node.x += self.path_resolution * math.cos(theta)
                    new_node.y += self.path_resolution * math.sin(theta)
                    new_node.path_x.append(new_node.x)
                    new_node.path_y.append(new_node.y)

                d, _ = self.calc_distance_and_angle(new_node, to_node)
                if d <= self.path_resolution:
                    new_node.path_x.append(to_node.x)
                    new_node.path_y.append(to_node.y)
                    new_node.x = to_node.x
                    new_node.y = to_node.y

                new_node.parent = from_node

                return new_node

            def generate_final_course(self, goal_ind):
                path = [[self.end.x, self.end.y]]
                node = self.node_list[goal_ind]
                while node.parent is not None:
                    path.append([node.x, node.y])
                    node = node.parent
                path.append([node.x, node.y])

                return path

            def calc_dist_to_goal(self, x, y):
                dx = x - self.end.x
                dy = y - self.end.y
                return math.hypot(dx, dy)

            def get_random_node(self):
                if random.randint(0, 100) > self.goal_sample_rate:
                    rnd = self.Node(
                        random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.min_rand, self.max_rand))
                else:  # goal point sampling
                    rnd = self.Node(self.end.x, self.end.y)
                return rnd

            @staticmethod
            def get_nearest_node_index(node_list, rnd_node):
                dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                         for node in node_list]
                minind = dlist.index(min(dlist))

                return minind

            @staticmethod
            def check_collision(node, static_obstacles):
                if node is None:
                    return False
                for obs in static_obstacles:
                    if LineString(list(zip(node.path_x, node.path_y))).intersects(obs.shape):
                        return False
                    if LineString(list(zip(node.path_x, node.path_y))).distance(obs.shape) < 3:
                        return False
                return True

            @staticmethod
            def calc_distance_and_angle(from_node, to_node):
                dx = to_node.x - from_node.x
                dy = to_node.y - from_node.y
                d = math.hypot(dx, dy)
                theta = math.atan2(dy, dx)
                return d, theta

        rrt_star = RRT(
            start=start,
            goal=goal,
            rand_area=rand_area,
            obstacle_list=obstacle_list,
            expand_dis=20)

        path = rrt_star.planning()
        if path is None:
            print('Cannot find path')
        else:
            print('Found Path')
            return path

    def __init__(self,
                 goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 sg: SpacecraftGeometry,
                 sp: SpacecraftParameters):
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.sg = sg
        self.sp = sp
        self.name = None

    def on_episode_init(self, my_name: PlayerName):
        self.name = my_name

    def orientate(self, sim_obs:  SimObservations, angle: float) -> SpacecraftCommands:
        acc = 0
        kd = 0.5
        dpsi = sim_obs.players[self.name].state.dpsi
        psi = sim_obs.players[self.name].state.psi
        if np.abs(psi-angle) > 0.1:
            acc = kd*(angle-psi)
        if acc > 0:
            return SpacecraftCommands(acc_left=0, acc_right=acc)
        else:
            return SpacecraftCommands(acc_left=acc, acc_right=0)

    def stabilize(self, sim_obs: SimObservations) -> SpacecraftCommands:
        vx = sim_obs.players[self.name].state.vx
        vy = sim_obs.players[self.name].state.vy
        dpsi = sim_obs.players[self.name].state.dpsi
        psi = sim_obs.players[self.name].state.psi
        heading_angle = np.arctan2(vy, vx)
        acc = 0
        if np.abs(dpsi) < 0.1 and np.abs(heading_angle-psi) < 0.1:
            kd = 0.5
            v_craft = np.hypot(vx, vy)
            if np.abs(v_craft) > 0.05:
                acc = -kd*v_craft
        else:
            self.orientate(sim_obs, heading_angle)
            print('Gros chien')

        return SpacecraftCommands(acc_left=acc, acc_right=acc)

    def export_path(self, path):
        if os.path.exists("path.txt"):
            os.remove("path.txt")
        writer = open("path.txt", "w")
        for element in path:
            writer.write(str(element[0]) + ', ' + str(element[1]) + "\n")
        writer.close()

    def import_path(self):
        initial_path=[]
        q = open('path.txt', 'r')
        for i in q.readlines():
            tmp = i.split(', ')
            initial_path.append((float(tmp[0]), float(tmp[1])))
        return initial_path

    def get_commands(self, sim_obs: SimObservations) -> SpacecraftCommands:
        """ This method is called by the simulator at each time step.

        This is how you can get your current state from the observations:
        my_current_state: SpacecraftState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        # Parameters
        goal_x = self.goal.goal.centroid.x
        goal_y = self.goal.goal.centroid.y
        start = [5, 5]
        goal = [goal_x, goal_y]
        rand_area = [0, 100]
        obstacle_list = self.static_obstacles
        spacecraft_path=[]
        expand_dis = 20
        '''if sim_obs.time == 0.0:
            spacecraft_path=self.path(start, goal, obstacle_list, rand_area)
            self.export_path(spacecraft_path)
        else:
            if os.path.exists("path.txt"):
                spacecraft_path = self.import_path()
            else:
                print('File not found, big error frerot')'''


        ctrl: SpacecraftCommands = SpacecraftCommands(acc_right=0, acc_left=0)
        if sim_obs.time < 3:
            ctrl = self.stabilize(sim_obs)
        if sim_obs.time > 8:
            ctrl.acc_left=10
            ctrl.acc_right=10'
        '''point2=spacecraft_path[-2]
        x = sim_obs.players[self.name].state.x
        y = sim_obs.players[self.name].state.y
        if sim_obs.time<6:
            self.stabilize(sim_obs)
        else:
            ctrl = self.orientate(sim_obs, np.arctan2((point2[1]-y),(point2[0]-x)))
        if sim_obs.time>10:
            ctrl.acc_left =10
            ctrl.acc_right=10'''


        return SpacecraftCommands(acc_left=ctrl.acc_left, acc_right=ctrl.acc_right)