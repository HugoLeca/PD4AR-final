import matplotlib
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from matplotlib import pyplot as plt
import numpy as np

#######added function to run a simulation
from copy import deepcopy
from decimal import Decimal as D
from typing import Optional

from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation

from dg_commons import PlayerName, DgSampledSequence
from dg_commons.maps.shapes_generator import create_random_starshaped_polygon
from dg_commons.planning import PlanningGoal
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent

from dg_commons.sim.models.obstacles import ObstacleGeometry, DynObstacleParameters
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel, DynObstacleState, DynObstacleCommands
from dg_commons.sim.models.spacecraft import SpacecraftModel, SpacecraftState
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from numpy import deg2rad
from shapely.geometry import Polygon, MultiPoint, Point

from scipy.stats import qmc

# imports for Halton points generation
from typing import Optional, Dict
from dg_commons.sim.models.obstacles import StaticObstacle
# points = MultiPoint([(0.0, 0.0), (1.0, 1.0)])

import math
import random

# indicate path to final21 folder
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/guillaumelecronier/Sanbox/src/pdm4ar/exercises_def/final21')

from agent import Pdm4arAgent

PDM4AR = PlayerName("PDM4AR")

########location of the file containing the basic scenario
from scenario import get_dgscenario
from scenario_test import get_dgscenario_test


def _get_sim_context_static(scenario: DgScenario, goal: PlanningGoal, x0: SpacecraftState) -> SimContext:
    model = SpacecraftModel.default(x0)
    models = {PDM4AR: model}
    missions = {PDM4AR: goal}
    players = {PDM4AR: Pdm4arAgent(
        static_obstacles=deepcopy(list(scenario.static_obstacles.values())),
        goal=goal,
        sg=deepcopy(model.get_geometry()),
        sp=deepcopy(model.sp))
    }

    return SimContext(
        dg_scenario=scenario,
        models=models,
        players=players,
        missions=missions,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"),
                            sim_time_after_collision=D(1), max_sim_time=D(10)),
    )


def get_sim_context_static(seed: Optional[int] = None, mode: int = 0) -> SimContext:
    # mode 0 --> original task scenario ; mode 1 --> custom scenario.Defaut = 0
    if mode == 0:
        dgscenario, goal, x0 = get_dgscenario(seed)
    if mode == 1:
        dgscenario, goal, x0 = get_dgscenario_test(seed)

    simcontext = _get_sim_context_static(dgscenario, goal, x0)
    simcontext.description = "static-environment"
    return simcontext


# code for creating a gif animation !!specify the animation path destination!!
def anim(sim_context: SimContext):
    fn = "/Users/guillaumelecronier/Sanbox/src/pdm4ar/exercises/final21/out_pictures/animation_demo.gif"
    create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 16), dt=50, dpi=120, plot_limits=None)
    return None


# Halton point generation

def Halton_points_generator(Map_X, Map_Y, number_points):
    sample_points = np.zeros((number_points, 2))

    sampler = qmc.Halton(d=2, scramble=False)
    sample_unity = sampler.random(n=number_points)

    sample_points[:, 0] = Map_X * sample_unity[:, 0]
    sample_points[:, 1] = Map_Y * sample_unity[:, 1]

    return sample_points


def sample_points_wout_obstacle(sample_points, static_obstacles: Dict[int, StaticObstacle]):
    resulting_points = sample_points
    index_drop = []
    for i in range(0, sample_points.shape[0]):
        p1 = Point(sample_points[i, :])
        for obstacle in static_obstacles.values():
            if p1.within(obstacle.shape):
                index_drop.append(i)
                print(f"point {p1} is in obstacle. Dropping point")

    resulting_points = np.delete(resulting_points, index_drop, axis=0)

    return resulting_points


"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

show_animation = True


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
                 expand_dis=0.1,
                 path_resolution=0.01,
                 goal_sample_rate=5,
                 max_iter=10000,
                 play_area=None
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
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
                    self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            '''if animation and i % 5 == 0:
                self.draw_graph(rnd_node)'''

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            '''if animation and i % 5:
                self.draw_graph(rnd_node)'''

        return None  # cannot find path

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

    '''def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)
'''
    '''@staticmethod'''
    '''def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)'''

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, static_obstacles):
        if node is None:
            return False

        for obs in static_obstacles:
            dx_list = [x for x in node.path_x]
            dy_list = [y for y in node.path_y]
            for i in range(0, np.shape(dx_list)[0]):
                p1 = Point(dx_list[0], dy_list[i])
                if p1.within(obs.shape):
                    return False

        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    dg_scenario, goal, _ = get_dgscenario()
    ax = plt.gca()
    shapely_viz = ShapelyViz(ax)

    for s_obstacle in dg_scenario.static_obstacles.values():
        # print(s_obstacle)
        shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)

    # Halton point genereation
    max_size = 100
    Npoints = 300
    Halton_Points = Halton_points_generator(max_size, max_size, Npoints)

    # check if point is inside polygon. Dropping the point if inside
    pts = sample_points_wout_obstacle(Halton_Points, dg_scenario.static_obstacles)
    print(f"initial size : {Halton_Points.shape}, resulting_size : {pts.shape}")
    # Points = np.array([(0.0, 1.0) for idx in range(8)])
    # print(Points)

    # print(MultiPoint(Points))  #Multipoint constructor takes a 1-D array of tuples
    # pts_wout_obs = sample_points_wout_obstacle(Points, dg_scenario.static_obstacles) #dg_scenario.static_obstacles is a Dict[int, StaticObstacle]
    # shapely_viz.add_shape()

    # shapely_viz.add_shape(Points, color = "orange")
    '''for point in Halton_Points:
        print(point)
        shapely_viz.add_shape(Point(point), radius = 0.2, color = 'white')'''

    print("start " + __file__)

    # ====Search Path with RRT====
    # [x, y, radius]$
    obstacleList = dg_scenario.static_obstacles.values()
    x0 = 10
    y0 = 10
    xg = 95
    yg = 95

    shapely_viz.add_shape(Point((x0, y0)), radius=0.2, color='red')
    shapely_viz.add_shape(Point((xg, yg)), radius=0.2, color='red')
    # Set Initial parameters
    rrt = RRT(
        start=[x0, y0],
        goal=[xg, yg],
        rand_area=[0, 100],
        obstacle_list=obstacleList,
        # play_area=[0, 10, 0, 14]
    )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            # rrt.draw_graph()
            for (x, y) in path:
                shapely_viz.add_shape(Point((x, y)), radius=0.2, color='white')

    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_facecolor('k')
    ax.set_aspect("equal")

    # getting the simulaton's context
    # seed = 0
    # sim_context = get_sim_context_static(seed, mode = 1)

    # #running the siulation and saving the logs
    # sim = Simulator()
    # sim.run(sim_context)

    # plotting the logs into a .gif animation
    # anim(sim_context)
    plt.show()