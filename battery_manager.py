import hassapi as hass

"""App to control batteries charging and discharging based on elecricity prices.

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
        self.enable_AC_input_entity = self.args["AC_input"]
        self.charge_control_entity = self.args["charge_control"]
        self.charge_state_entity = self.args["charge_state"]
        self.max_charge = self.args.get("max_charge", 90)
        self.min_charge = self.args.get("max_charge", 30)
        self.emergency_charge = self.args.get("emergency_charge", 10)
