from typing import Sequence

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands
<<<<<<< HEAD
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry, SpacecraftParameters
from shapely.affinity import translate
=======
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry

# Imported libraries
from scipy.stats import qmc
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KDTree


def Halton_points_generator(Map_X, Map_Y, number_points):
    sample_points = np.zeros((number_points, 2))

    sampler = qmc.Halton(d=2, scramble=False)
    sample_unity = sampler.random(n=number_points)

    sample_points[:, 0] = Map_X * sample_unity[:, 0]
    sample_points[:, 1] = Map_Y * sample_unity[:, 1]

    return sample_points


def sample_points_wout_obstacle(sample_points, static_obstacles: Sequence[StaticObstacle]):
    resulting_points = sample_points
    index_drop = []
    for i in range(0, sample_points.shape[0]):
        p1 = Point(sample_points[i, :])
        if p1.within(static_obstacles):
            index_drop.append(i)

    resulting_points = np.delete(resulting_points, index_drop, axis=0)

    return resulting_points


def measure_distance(point_A, point_B):
    distance = np.sqrt(
        np.square(point_A[0] - point_B[0]) + np.square(point_A[1] - point_B[1]))

    return distance


def nearest_neighbor(sample_points):
    tree = KDTree(sample_points)
    nearest_dist, nearest_ind = tree.query(sample_points, k=2)

    return nearest_dist, nearest_ind


def collision_checker(Map, obstacles: Sequence[StaticObstacle], sample_points):
    for i in range(len(obstacles)):
        shape = obstacles[i].shape._geom
        geometry = obstacles[i].geometry
    return None


def Intersection(line, center, radius):
    a = np.dot(line.dirn, line.dirn)
    b = 2 * np.dot(line.dirn, line.p - center)
    c = np.dot(line.p - center, line.p - center) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
        return False

    return True


def local_planner(Map_X, Map_Y, sample_points, static_obstacles: Sequence[StaticObstacle]):
    nearest_dist, nearest_ind = nearest_neighbor(sample_points)
    nearest_ind_init = nearest_ind[0, 1]
    visited = [0, nearest_ind_init]
    distance_lim = 0
    bibli_distance = []

    while (visited.shape[0] != sample_points.shape[0]):
        nearest_dist, nearest_ind = nearest_neighbor(sample_points)

        for i in range(0, visited.shape[0]):
            nearest_ind_init = nearest_ind[visited[i], 1]
            distance_cons = nearest_dist[visited[i], 1]
            if distance_cons >= distance_lim:
                distance_lim = distance_cons
                choosen_ind = nearest_ind[visited[i], 1]
        visited.append(choosen_ind)
        bibli_distance = []  # STOCKAGE

        sample_points = np.delete(sample_points, nearest_ind_init, axis=0)
        visited.append()

    return None


def shortest_path(AdjencyMap):
    return None

>>>>>>> ce77930737efa300564ce2b6e495f72cb02fdbea

class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do NOT modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 sg: SpacecraftGeometry,
                 sp: SpacecraftGeometry):
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.sg = sg
        self.sp = sp
        self.name = None

    def on_episode_init(self, my_name: PlayerName):
        self.name = my_name

    def get_commands(self, sim_obs: SimObservations) -> SpacecraftCommands:
        """ This method is called by the simulator at each time step.

        This is how you can get your current state from the observations:
        my_current_state: SpacecraftState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        # Parameters
        Map_X = 100
        Map_Y = 100
        number_points = 100

        sample_points = Halton_points_generator(Map_X, Map_Y, number_points)
        sample_points = sample_points_wout_obstacle(sample_points, self.static_obstacles)

        return SpacecraftCommands(acc_left=1, acc_right=1)