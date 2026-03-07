# Hyperparameters

# Maximum number of chargers per station
STATION_CHARGER_LIMIT = 8 # m

# The maximum number of robots each charger can serve
CHARGER_ROBOT_LIMIT = 2 # q

CAPACITY_LIMIT = STATION_CHARGER_LIMIT * CHARGER_ROBOT_LIMIT

# Build cost
BUILD_COST = 5000 # c_b

# Human rescue cost
RESCUE_COST = 1000 # c_h

# Maintenance cost
MAINTAIN_COST = 500 #  c_m

# Charging cost per km
CHARGE_COST = 0.42 # c_c

# Stochastic & Range parameters
LAMBDA = 0.012
R_MIN = 10
R_MAX = 175

# import this file using "from Hyperparameter import * "