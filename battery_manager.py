import hassapi as hass
import datetime
import numpy as np
import pandas as pd
from astar import AStar

"""App to control batteries charging and discharging based on elecricity prices.

It depends on another app publishing a Pandas time series of current and future
energy as a global variable "electricity_prices".

Configuration:

AC_input : Switch entity controlling the AC input to the battery
charge_control : Numerical entity controlling the target charge state in %
charge_state : Numerical sensor reporting the charge state in %
max_charge : Maximum charge target to set, optional, default: 90
max_charge : Minimum charge target to set, optional, default: 30
emergency_charge : Charge state at which the AC input will be enabled no matter what, optional, default: 10
max_charge_rate : Maximum achievable charge rate with AC charging in % per hour, optional, default: 15
mean_discharge_rate : Mean assumed discharge rate in % per hour, optional, default: 10
round_trip_efficiency : Round trip efficiency of charge-discharge cycle, optional, default: 0.8
publish : Variable name to publish charge plan to, optional, default: ""

"""


class BatteryManager(hass.Hass):
    async def initialize(self):
        # Set configuration
        self.enable_AC_input_entity = self.args["AC_input"]
        self.charge_control_entity = self.args["charge_control"]
        self.charge_state_entity = self.args["charge_state"]
        self.max_charge = int(self.args.get("max_charge", 90))
        self.min_charge = int(self.args.get("min_charge", 30))
        self.emergency_charge = int(self.args.get("emergency_charge", 10))
        self.max_charge_rate = float(self.args.get("max_charge_rate", 15))
        self.mean_discharge_rate = float(self.args.get("mean_discharge_rate", 10))
        self.round_trip_efficiency = float(self.args.get("round_trip_efficiency", 0.8))
        self.publish = self.args.get("publish", "")

        self.log(
            "Loaded with configuration: %s"
            % (
                {
                    "Max Charge": self.max_charge,
                    "Min Charge": self.min_charge,
                    "Emergency Charge": self.emergency_charge,
                    "Max Charge Rate": self.max_charge_rate,
                    "Mean Discharge Rate": self.mean_discharge_rate,
                    "Round-Trip Efficiency": self.round_trip_efficiency,
                    "Publish": self.publish,
                },
            )
        )

        # Are we in emergency charge mode?
        self.emergency = False

        # Update battery state every minute
        self.run_minutely(self.control_battery, datetime.time(minute=0, second=1))

    async def control_battery(self, kwargs):
        prices = self.global_vars["electricity_prices"]

        now = pd.Timestamp.now(tz=prices.index.tz)
        current_price = prices.asof(now)

        # Emergency charge
        threshold = self.emergency_charge
        charge = int(await self.get_state(self.charge_state_entity))
        if charge < threshold:
            if not self.emergency:
                self.log("Entering emergency charge state!")
            self.emergency = True

        estimator = LookupEstimator(initial_guess=self.mean_discharge_rate)

        # Use AStar algorithm to find cheapest way
        astar = AStarStrategy(
            prices=prices,
            max_charge_rate=self.max_charge_rate,
            consumption_estimator=estimator,
            round_trip_efficiency=self.round_trip_efficiency,
            min_charge=self.min_charge,
            max_charge=self.max_charge,
        )
        current_state = (now, charge)
        end = prices.index[-1] + pd.Timedelta(hours=1)
        target_state = (end, self.min_charge)
        steps = list(astar.astar(current_state, target_state))

        self.log(estimation.discharge_dict)

        # Publish plan for other apps to use
        if self.publish:
            index, values = zip(*steps)
            series = pd.Series(values, index)
            self.global_vars[self.publish] = series

        if len(steps) < 2:
            # Something has gone wrong.
            self.log("Less than two states in the optimal path!")
            self.store()
            return

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
            # Only switch to "store" if next step is longer than 10 minutes in the future.
            # Otherwise we would always switch to "store" towards when reaching the target level.
            # This would mess up discharging.
            if (steps[1][0] - steps[0][0]) > pd.Timedelta(minutes=10):
                # Use 'charge' with target instead of 'store' in case the current charge level
                # has changed while we were calculating the optimal route
                await self.charge(next_charge)

    async def charge(self, target=None):
        """Charge the battery."""

        if target is None:
            target = self.max_charge

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

    """

    def __init__(self, initial_guess, split_by=("hour",)):
        self.discharge_dict = {}
        self.initial_guess = initial_guess
        self.split_by = split_by

    def get_discharge_rate(self, time):
        """Get the expected discharge rate at the given time.

        If the given time is not in the dictionary yet, it will be added.
        """

        key = tuple(getattr(time, x)() for x in self.split_by)
        rate = self.discharge_dict.get(key, self.initial_guess)
        self.discharge_dict[key] = rate
        return rate

    def __call__(self, t1, t2):
        """Estimate the consumption between the two given times."""

        sample_points = ps.date_range(start=t1, stop=t2, periods=20, include="left")
        rates = sample_points.map(self.get_discharge_rate)
        mean_rate = rates.mean()
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
    max_charge : The maximum charge state to aum for

    """

    def __init__(
        self,
        prices,
        max_charge_rate,
        consumption_estimator,
        round_trip_efficiency=1.0,
        min_charge=30,
        max_charge=90,
    ):
        super().__init__()

        self.prices = prices
        self.max_charge_rate = max_charge_rate
        self.consumption_estimator = consumption_estimator
        self.round_trip_efficiency = round_trip_efficiency
        self.min_charge = min_charge
        self.max_charge = max_charge

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

        cost = consumption * min_price
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
        max_charge = int(node[1] + time_diff * self.max_charge_rate)

        charges = set(range(min_charge, max_charge, 2))  # Limit number of choices
        charges.add(max_charge)  # Make sure max is in there
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
                yield (next_time, c)
