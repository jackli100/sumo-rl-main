"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces

class Timer:
    def __init__(self):
        self.time_elapsed = 0
    
    def start(self):
        self.time_elapsed = 0
    
    def reset(self, reset_value=0):
        self.time_elapsed = reset_value
    
    def update(self):
        self.time_elapsed += 1
    
    def elapsed_time(self):
        return self.time_elapsed
    
class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time_1 = yellow_time
        self.yellow_time_2 = yellow_time
        self.red_time = 1
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.green_phase_state = 'rrrrrrrrrrrr' # 如果确认要换相位，就把这个值改成新相位的状态
        self.new_phase_state = 'rrrrrrrrrrrr' # 先放进去这个值，不一定要换
        self.timers = [Timer() for _ in range(12)]
        self.is_yellow_1 = False
        self.is_yellow_2 = False
        self.is_red = False
        self.is_red = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo
        self.old_green_phase = 0

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    # D:\trg1vr\sumo-rl-main\sumo-rl-main\sumo_rl\environment\env.py

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        self.red_dict = {}

        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))

        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()
        self.green_phase_state = self.all_phases[0].state

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state_1 = ""
                yellow_state_2 = ""
                red_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state_1 += "y"
                        yellow_state_2 += "r"
                        red_state += "r"
                    elif (p1.state[s] == "r" or p1.state[s] == "s") and (p2.state[s] == "G" or p2.state[s] == "g"):
                        yellow_state_1 += "r"
                        yellow_state_2 += "u"
                        red_state += "r"
                    else:
                        yellow_state_1 += p1.state[s]
                        yellow_state_2 += p1.state[s]
                        red_state += p1.state[s]

                self.yellow_dict[(i, j)] = (len(self.all_phases), len(self.all_phases) + 1, len(self.all_phases) + 2) # 储存了所有黄色相位+红色相位的索引
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time_1, yellow_state_1))
                self.all_phases.append(self.sumo.trafficlight.Phase(self.red_time, red_state))
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time_2, yellow_state_2))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)


    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        for i in range(len(self.timers)):
            self.timers[i].update()

        if self.is_yellow_1 and self.time_since_last_phase_change == self.yellow_time_1:
            red_phase_index = self.yellow_dict[self.old_green_phase, self.green_phase][1]
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[red_phase_index].state)
            self.is_yellow_1 = False
            self.is_red = True

        if self.is_red and self.time_since_last_phase_change == self.yellow_time_1 + self.red_time:
            yellow_phase_2_index = self.yellow_dict[self.old_green_phase, self.green_phase][2]
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[yellow_phase_2_index].state)
            self.is_red = False
            self.is_yellow_2 = True

        if self.is_yellow_2 and self.time_since_last_phase_change == self.yellow_time_1 + self.red_time + self.yellow_time_2:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow_2 = False


    def set_next_phase(self, new_phase: int):
        '''
        确定要换之前：
            new_phase 是新的相位
            green_phase 是旧的相位
            这对关系用于寻找对应的黄色相位和红色相位
            
        确定要换之后：
            old_green_phase 是旧的相位，green_phase 是新的相位, 这对关系用于寻找对应的黄色相位和红色相位
        '''
        new_phase = int(new_phase)
        # new_phase_state 就是一个过渡用的变量，如果确认要换相位，就把这个值改成新相位的状态

        if self.green_phase == new_phase:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time

        elif not self.can_switch():
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
            
        else: # 确认要换
            self.new_phase_state = self.all_phases[new_phase].state       
            yellow_phase_index = self.yellow_dict[self.green_phase, new_phase][0]
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[yellow_phase_index].state)
            self.old_green_phase = self.green_phase            
            self.green_phase = new_phase
            # green_phase_state 目前是旧的相位，new_phase_state是新的相位
            self.update_lights()
            self.green_phase_state = self.new_phase_state
            self.next_action_time = self.env.sim_step + self.yellow_time_1 + self.yellow_time_2 + self.red_time

            self.is_yellow_1 = True
            self.time_since_last_phase_change = 0
            

    def can_switch(self):
        for i in range(len(self.green_phase_state)):
            if self.green_phase_state[i] in ('G', 'g'):
                if self.timers[i].elapsed_time() < self.min_green:
                    return False
        return True

    def update_lights(self):
        '''
        green_phase_state 是旧的相位，new_phase_state 是新的相位
        '''
        for i in range(len(self.green_phase_state)):
            if self.new_phase_state[i] in ('G', 'g'):
                if self.green_phase_state[i] not in ('G', 'g'):
                    self.timers[i].reset(-self.yellow_time_1 - self.yellow_time_2 - self.red_time)


    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time_1 + self.yellow_time_2 + self.red_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }
