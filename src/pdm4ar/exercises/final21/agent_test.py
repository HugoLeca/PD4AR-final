import matplotlib
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from matplotlib import pyplot as plt


#######added function to run a simulation
from copy import deepcopy
from decimal import Decimal as D
from typing import Optional

from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from reprep import Report, MIME_GIF


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
from shapely.geometry import Polygon



# indicate path to final21 folder
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/hugol/OneDrive/Documents/ETHZ/PD4AR/PD4AR-final/src/pdm4ar/exercises_def/final21')

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
    #mode 0 --> original task scenario ; mode 1 --> custom scenario.Defaut = 0
    if mode == 0:
        dgscenario, goal, x0 = get_dgscenario(seed)
    if mode == 1:
        dgscenario, goal, x0 = get_dgscenario_test(seed)

    
    simcontext = _get_sim_context_static(dgscenario, goal, x0)
    simcontext.description = "static-environment"
    return simcontext



#code for creating a gif animation !!specify the animation path destination!!
def anim(sim_context: SimContext):
    fn = "C:/Users/hugol/OneDrive/Documents/ETHZ/PD4AR/PD4AR-final/src/pdm4ar/exercises/final21/out_pictures/animation_test.gif"
    create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 16), dt=50, dpi=120, plot_limits=None)
    return None



if __name__ == '__main__':
    matplotlib.use('TkAgg')
    dg_scenario, goal, _ = get_dgscenario()
    ax = plt.gca()
    shapely_viz = ShapelyViz(ax)

    for s_obstacle in dg_scenario.static_obstacles.values():
        shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)

    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_facecolor('k')
    ax.set_aspect("equal")

    #getting the simulaton's context
    seed = 0
    sim_context = get_sim_context_static(seed, mode = 1)

    #running the siulation and saving the logs
    sim = Simulator()
    sim.run(sim_context)

    #plotting the logs into a .gif animation
    anim(sim_context)
    #plt.show()