import hassapi as hass
import datetime
import numpy as np
import pandas as pd
from astar import AStar
from functools import lru_cache
import pickle
import os

"""App to control batteries charging and discharging based on elecricity prices.

It depends on another app publishing a Pandas time series of current and future
energy as a global variable "electricity_prices".

Configuration:

AC_input : Switch entity controlling the AC input to the battery
charge_control : Numerical entity controlling the target charge state in %
charge_state : Numerical sensor reporting the charge state in %
max_charge : Maximum charge target to set, optional, default: 90
min_charge : Minimum charge target to set, optional, default: 30
undershoot_charge : Minimum charge to discharge to, optional, default: 30
emergency_charge : Charge state at which the AC input will be enabled no matter what, optional, default: 10
max_charge_rate : Maximum achievable charge rate with AC charging in % per hour, optional, default: 15
alt_max_charge_rate : Alternative maximum achievable charge rate with AC charging in % per hour, enabled with alt_rate_enable switch entity, optional, default: 15
alt_rate_control : Switch entity to enable the alternative charge rate, optional
mean_discharge_rate : Mean assumed discharge rate in % per hour, optional, default: 10
round_trip_efficiency : Round trip efficiency of charge-discharge cycle, optional, default: 0.8
publish : Variable name to publish charge plan to, optional, default: ""
save_file : Path to file to store statistics for better predictionsa, optional, default: ""
learning_attributes : Attributes and methods of the `datetime` to use to learn better predictions,
    optional, default: ['weekday', 'hour']
learning_factor : Speed at which to update assumed energy consumption with measurements,
    optional, default: 0.05
enable_control: Entity used to switch on or off the battery control, optional

"""


class BatteryManager(hass.Hass):
    async def initialize(self):
        # Unpin the app, so long Astar calculations do not block other callbacks
        await self.set_app_pin(False)

        # Set configuration
        self.enable_AC_input_entity = self.args["AC_input"]
        self.enable_control_entity = self.args.get("enable_control", None)
        self.charge_control_entity = self.args["charge_control"]
        self.charge_state_entity = self.args["charge_state"]
        self.max_charge = int(self.args.get("max_charge", 90))
        self.min_charge = int(self.args.get("min_charge", 30))
        self.undershoot_charge = int(self.args.get("undershoot_charge", 30))
        self.emergency_charge = int(self.args.get("emergency_charge", 10))
        self.max_charge_rate = float(self.args.get("max_charge_rate", 15))
        self.alt_max_charge_rate = float(self.args.get("alt_max_charge_rate", 15))
        self.enable_alt_rate_entity = self.args.get("alt_rate_control", "")
        self.mean_discharge_rate = float(self.args.get("mean_discharge_rate", 10))
        self.round_trip_efficiency = float(self.args.get("round_trip_efficiency", 0.8))
        self.publish = self.args.get("publish", "")
        self.save_file = self.args.get("save_file", "")
        self.learning_attributes = self.args.get(
            "learning_attributes", ["weekday", "hour"]
        )
        self.learning_factor = self.args.get("learning_factor", 0.05)

        self.log(
            "Loaded with configuration: %s"
            % (
                {
                    "Max Charge": self.max_charge,
                    "Min Charge": self.min_charge,
                    "Undershoot Charge": self.undershoot_charge,
                    "Emergency Charge": self.emergency_charge,
                    "Max Charge Rate": self.max_charge_rate,
                    "Mean Discharge Rate": self.mean_discharge_rate,
                    "Round-Trip Efficiency": self.round_trip_efficiency,
                    "Publish": self.publish,
                    "Save File": self.save_file,
                    "Learning Attributes": self.learning_attributes,
                    "Learning Factor": self.learning_factor,
                },
            )
        )

        # Are we in emergency charge mode?
        self.emergency = False

        # Charge state last time we looked
        self.last_charge = int(await self.get_state(self.charge_state_entity))
        prices = self.global_vars.get("electricity_prices", None)
        self.log(prices)
        if prices is None or len(prices) == 0:
            self.last_time = pd.Timestamp.now(tz="UTC")
        else:
            self.last_time = pd.Timestamp.now(tz=prices.index.tz)

        # Estimator of future energy consumption
        self.estimator = LookupEstimator(
            initial_guess=self.mean_discharge_rate,
            split_by=self.learning_attributes,
            update_speed=self.learning_factor,
        )
        if self.save_file and os.path.exists(self.save_file):
            try:
                self.estimator.load_stats(self.save_file)
            except Exception as e:
                self.log(f"Failed to load stats from file {self.save_file}")
                self.log(e)

        # Update battery state every 5 minutes, starting 30 seconds after the full hour
        now = datetime.datetime.now()
        start = now.replace(minute=0, second=30)
        while (start - now).total_seconds() <= 0:
            start += datetime.timedelta(minutes=5)
        self.run_every(
            self.control_battery,
            start=start,
            interval=60 * 5,
        )

        # Check for an emergency more frequently
        self.run_minutely(self.check_emergency, datetime.time(hour=0))

        # Also additionally check for emergencies on charge state changes
        entity = self.get_entity(self.charge_state_entity)
        entity.listen_state(self.charge_change_callback)

    async def charge_change_callback(self, entity, attribute, old, new, kwargs):
        """Called whenever the charge state changes."""
        await self.check_emergency({})

    async def is_discharging(self):
        """Check whether the current state is discharging the battery."""

        state = await self.get_state(self.enable_AC_input_entity)
        return state == "off"

    async def check_emergency(self, kwargs):
        """Check that the charge is not below the emergency level.

        Start charging if it is.
        """

        charge = int(await self.get_state(self.charge_state_entity))
        threshold = self.emergency_charge
        if charge < threshold:
            if not self.emergency:
                self.log("Entering emergency charge state!")
            self.emergency = True
            await self.store()

    async def control_battery(self, kwargs):
        """Callback function to actually control the battery."""

        # Do nothing if battery control is swtiched off
        if (
            self.enable_control_entity is not None
            and await self.get_state(self.enable_control_entity) == "off"
        ):
            return

        prices = self.global_vars.get("electricity_prices", None)
        if prices is None or len(prices) == 0:
            # No prices? nothing to do
            self.log("No prices available! Switching to store mode.")
            await self.store()
            return

        now = pd.Timestamp.now(tz=prices.index.tz)
        current_price = prices.asof(now)
        charge = int(await self.get_state(self.charge_state_entity))
        target = int(await self.get_state(self.charge_control_entity))
        charge_diff = charge - self.last_charge
        time_diff = (now - self.last_time).total_seconds() / 3600  # in hours

        if await self.is_discharging():
            # Learn how fast we discharge
            self.estimator.learn_discharge_rate(
                self.last_time, -charge_diff / time_diff
            )
            if self.save_file:
                self.estimator.save_stats(self.save_file)
        elif charge < target:
            # We are probably charging
            # Slowly learn how fast we charge
            self.max_charge_rate *= 0.999
            self.max_charge_rate += 0.001 * charge_diff / time_diff

        self.last_charge = charge
        self.last_time = now

        # Use AStar algorithm to find cheapest way
        astar = AStarStrategy(
            prices=prices,
            max_charge_rate=self.max_charge_rate,
            consumption_estimator=self.estimator,
            round_trip_efficiency=self.round_trip_efficiency,
            min_charge=self.min_charge,
            undershoot=self.undershoot_charge,
            max_charge=self.max_charge,
            debug=self.log,
        )
        current_state = (now, charge)
        end = prices.index[-1] + pd.Timedelta(hours=1)
        target_state = (end, self.min_charge)
        steps = list(
            await self.run_in_executor(astar.astar, current_state, target_state)
        )

        # Publish plan for other apps to use
        if self.publish:
            index, values = zip(*steps)
            series = pd.Series(values, index)
            self.global_vars[self.publish] = series

        if len(steps) < 2:
            # Something has gone wrong.
            self.log("Less than two states in the optimal path!")
            await self.store()
            return

        next_step = steps[1]
        self.set_charge_rate(current_state, next_step)

        next_charge = steps[1][1]
        if next_charge > charge:
            self.emergency = False
            await self.charge(next_charge)
        elif self.emergency:
            # Still in emergency state:
            await self.store()
        elif next_charge < charge:
            await self.discharge()
        else:
            # Use 'charge' with target instead of 'store' in case the current charge level
            # has changed while we were calculating the optimal route
            await self.charge(next_charge)

    async def set_charge_rate(self, now_state, target_state):
        """Set the alternative charge rate if appropriate."""

        if not self.enable_alt_rate_entity:
            # Nothing we can do
            return

        charge_diff = target_state[1] - now_state[1]
        time_diff = (target_state[0] - now_state[0]).total_seconds() / 3600  # in hours

        target_rate = charge_diff / time_diff

        if self.alt_max_charge_rate <= target_rate <= self.max_charge_rate:
            await self.turn_off(self.enable_alt_rate_entity)
        elif self.alt_max_charge_rate >= target_rate >= self.max_charge_rate:
            await self.turn_on(self.enable_alt_rate_entity)
        elif target_rate <= self.alt_max_charge_rate <= self.max_charge_rate:
            await self.turn_on(self.enable_alt_rate_entity)
        elif target_rate <= self.max_charge_rate <= self.alt_max_charge_rate:
            await self.turn_off(self.enable_alt_rate_entity)

    async def charge(self, target=None):
        """Charge the battery."""

        if target is None:
            target = self.max_charge
        if target < self.min_charge:
            target = self.min_charge

        await self.set_value(self.charge_control_entity, target)
        await self.turn_on(self.enable_AC_input_entity)

    async def store(self):
        """Keep current charge level.

        Just a convenience function to charge up to the current level.
        """

        target = int(await self.get_state(self.charge_state_entity))
        if target < self.min_charge:
            target = self.min_charge
        await self.charge(target)

    async def discharge(self):
        """Discharge the battery."""

        await self.turn_off(self.enable_AC_input_entity)


class LookupEstimator:
    """Look up expected consumption rates in table.

    Arguments
    ---------

    initial_guess : Discharge rate to populate the table with
    split_by : What datetime attributes to use for distinction
    update_speed : Relative weight of each new consumption measurement
    debug : Function used to log debug messages

    """

    def __init__(self, initial_guess, split_by=("hour",), update_speed=0.1, debug=None):
        self.discharge_dict = {}
        self.initial_guess = initial_guess
        self.split_by = tuple(split_by)
        self.update_speed = update_speed
        if debug is None:

            def debug(message):
                pass

        self.debug = debug

    def load_stats(self, filename):
        """Load stats from file."""

        with open(filename, "rb") as f:
            self.discharge_dict = pickle.load(f)

    def save_stats(self, filename):
        """Save stats to file."""

        with open(filename, "wb") as f:
            pickle.dump(self.discharge_dict, f)

    @lru_cache()
    def get_key(self, time):
        """Build the dictionary key from a given time."""
        key = []
        for attr in self.split_by:
            # Handle both attributes and methods
            try:
                val = getattr(time, attr)()
            except TypeError:
                val = getattr(time, attr)
            key.append(val)
        key = tuple(key)
        return key

    def get_discharge_rate(self, time):
        """Get the expected discharge rate at the given time.

        If the given time is not in the dictionary yet, it will be added.
        """

        key = self.get_key(time)
        rate = self.discharge_dict.get(key, self.initial_guess)
        self.discharge_dict[key] = rate
        return rate

    def learn_discharge_rate(self, time, rate):
        """Use the given measurement to improve future estimates."""

        key = self.get_key(time)
        old_rate = self.discharge_dict.get(key, self.initial_guess)
        new_rate = (1.0 - self.update_speed) * old_rate + self.update_speed * rate
        if new_rate < 1.0:
            # Make sure we assume a minimum amount of discharge
            # Otherwise the algorithms might break
            new_rate = 1.0
        self.discharge_dict[key] = new_rate

        # Clear cache since now values can be different
        self.__call__.cache_clear()

    @lru_cache()
    def __call__(self, t1, t2):
        """Estimate the consumption between the two given times."""

        sample_points = pd.date_range(start=t1, end=t2, freq="H", inclusive="left")
        self.debug(
            f"""Estimating consumption between {t1} and {t2}.
            Using sample points:
            {sample_points}"""
        )
        rates = sample_points.map(self.get_discharge_rate)
        mean_rate = rates.array.mean()
        time_diff = (t2 - t1).total_seconds() / 3600  # in hours

        return mean_rate * time_diff


class AStarStrategy(AStar):
    """Use A* algorithm to find the cheapest way of charging and discharging the battery.

    Arguments
    ---------

    prices : Time series of electricity prices, used for transition cost estimation
    max_charge_rate : Maximum charge rate of battery in % per hour
    consumption_estimator : Estimator that predicts the expected consumption (in %)
    round_trip_efficiency : Assumed round trip efficiency of charge-discharge cycle
    min_charge : The minimum charge state to aim for
    max_charge : The maximum charge state to aim for
    undershoot : Minimum charge state to discharge to
    debug : Function to log debug messages, optional

    """

    def __init__(
        self,
        prices,
        max_charge_rate,
        consumption_estimator,
        round_trip_efficiency=1.0,
        min_charge=30,
        max_charge=90,
        undershoot=30,
        debug=None,
        alt_max_charge_rate=None,
    ):
        super().__init__()

        self.prices = prices
        self.max_charge_rate = max_charge_rate
        if alt_max_charge_rate is None:
            alt_max_charge_rate = max_charge_rate
        self.alt_max_charge_rate = alt_max_charge_rate
        self.consumption_estimator = consumption_estimator
        self.round_trip_efficiency = round_trip_efficiency
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.undershoot = undershoot

        if debug is None:

            def debug(message):
                pass

        self.debug = debug

        # Pre-calculate minimum prices for rest of time range
        self.min_future_prices = prices[::-1].expanding().min()[::-1]

    def distance_between(self, n1, n2):
        """Calculate the cost when transitioning from state n1 to state n2."""

        charge_diff = n2[1] - n1[1]
        price = self.prices.asof(n1[0])
        consumption = self.consumption_estimator(n1[0], n2[0])

        if charge_diff < 0.0:
            # Negative charge diff means we use battery charge instead of direct AC
            direct_consumption = consumption + charge_diff
            if direct_consumption < 0.0:
                direct_consumption = 0.0
            charge_consumption = 0.0
        else:
            # Positive charge diff means we use AC and also charge the battery
            direct_consumption = consumption
            # Consider inefficiency of battery here
            charge_consumption = charge_diff / self.round_trip_efficiency

        cost = price * (direct_consumption + charge_consumption)
        return cost

    def heuristic_cost_estimate(self, current, goal):
        """What is the minimum cost to get from current to goal?"""

        # Get minimum price
        min_price = self.min_future_prices.asof(current[0])
        time_diff = (goal[0] - current[0]).total_seconds() / 3600  # in hours
        charge_diff = goal[1] - current[1]
        consumption = self.consumption_estimator(current[0], goal[0])

        if time_diff < 0.0:  # Travel to the past? I do not think so.
            return np.inf

        if charge_diff > 0.0:
            consumption += charge_diff / self.round_trip_efficiency
        else:
            consumption += charge_diff

        if consumption < 0.0:
            consumption = 0.0

        # Degrade the estiamte to make sure it is always lower than the real cost.
        cost = consumption * min_price * 0.9
        return cost

    def neighbors(self, node):
        """Return possible states reachable from node."""

        time = node[0]

        # Are we past the know price range? Nowhere to go!
        if time >= self.prices.index[-1] + pd.Timedelta(hours=1):
            return

        next_time = time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(
            hours=1
        )
        time_diff = (next_time - time).total_seconds() / 3600  # in hours
        consumption = self.consumption_estimator(time, next_time)
        min_charge = int(node[1] - consumption)
        if min_charge < 0:
            min_charge = 0
        max_charge = int(node[1] + time_diff * self.max_charge_rate)
        alt_max_charge = int(node[1] + time_diff * self.alt_max_charge_rate)
        if alt_max_charge > max_charge:
            # Make sure max_charge is the larger one
            max_charge, alt_max_charge = alt_max_charge, max_charge

        if max_charge > 100:
            max_charge = 100

        if alt_max_charge > 100:
            alt_max_charge = 100

        # Limit number of choices by 5 minute resolution
        charge_step = int(self.max_charge_rate / 12)  # 12 5 minute steps per hour
        discharge_step = int(consumption / time_diff / 12)
        if charge_step < 1:
            charge_step = 1
        if discharge_step < 1:
            discharge_step = 1
        charges = set(range(node[1], min_charge, -discharge_step))
        charges.update(range(node[1], max_charge, charge_step))
        charges.add(max_charge)  # Make sure max is in there
        charges.add(alt_max_charge)  # Make sure alt max is in there
        charges.add(min_charge)  # Make sure min is in there
        if self.min_charge >= min_charge:  # Make sure total min is in there
            charges.add(self.min_charge)
        if self.max_charge <= max_charge:  # Make sure total max is in there
            charges.add(self.max_charge)
        charges.add(node[1])  # Make sure current charge and neighbours are in there
        charges.add(node[1] + 1)  # This ensures we do not always switch to "store"
        charges.add(node[1] - 1)  # towards the end of full hours

        # Only yield acceptable charges
        for c in charges:
            if c >= self.min_charge and c <= self.max_charge:
                # Regular charge target
                yield (next_time, c)
            elif c == min_charge and c >= self.undershoot and c <= self.max_charge:
                # Allow uninterrupted discharge down to undershoot level
                if c == node[1]:
                    # Do not allow remaining at undershoot level
                    continue
                yield (next_time, c)
            elif c in (max_charge, alt_max_charge) and c <= self.max_charge:
                # Always allow uninterrupted charge
                yield (next_time, c)
