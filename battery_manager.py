import hassapi as hass
import datetime
import pandas as pd

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

"""


class BatteryManager(hass.Hass):
    def initialize(self):
        # Set configuration
        self.enable_AC_input_entity = self.args["AC_input"]
        self.charge_control_entity = self.args["charge_control"]
        self.charge_state_entity = self.args["charge_state"]
        self.max_charge = int(self.args.get("max_charge", 90))
        self.min_charge = int(self.args.get("min_charge", 30))
        self.emergency_charge = int(self.args.get("emergency_charge", 10))

        # Update battery state every minute
        self.run_minutely(self.control_battery, datetime.time(minute=0, second=1))

    def control_battery(self, kwargs):
        prices = self.global_vars["electricity_prices"]

        now = pd.Timestamp.now(tz=prices.index.tz)
        now = now.replace(minute=0, second=0, microsecond=0)
        current_price = prices[now]

        # Emergency charge
        emergency = self.emergency_charge
        charge = int(self.get_state(self.charge_state_entity))
        self.log((emergency, charge))
        if charge < emergency:
            self.charge(target=self.min_charge)
            return

        # Simple algorithm:
        # Charge if price is in lowest quartile,
        # Dischare in highers quartile,
        # Store otherwise

        if current_price < prices.quantile(0.25):
            self.charge()
        elif current_price > prices.quantile(0.75):
            self.discharge()
        else:
            self.store()

    def charge(self, target=None):
        """Charge the battery."""

        if target is None:
            target = self.max_charge

        self.set_state(self.charge_control_entity, state=target)
        self.turn_on(self.enable_AC_input_entity)

    def store(self):
        """Keep current charge level."""

        target = int(self.get_state(self.charge_state_entity))
        self.set_state(self.charge_control_entity, state=target)
        self.turn_on(self.enable_AC_input_entity)

    def discharge(self):
        """Discharge the battery."""

        self.turn_off(self.enable_AC_input_entity)
