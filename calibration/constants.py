

GENETIC_SOLUTIONS_FOLDER = 'genetic_solutions'
PARQUET_DATA_PATH = 'qr_takehome'

NORMAL_EQS_FOLDER = 'normal_eqs'
N_CV = 8

# normal equations are generated for N_DAY_TIME many day splits
N_DAY_TIME = 4 

# the training happens for CHOSEN_N_DAY_TIME many day splits
# it has to be a divisor of N_DAY_TIME
CHOSEN_N_DAY_TIME = 2

# names of the components in normal equations
NORMAL_EQ_COMPS = 'xx xy yy'.split()


HYPER_PARAM_LOGS = 'hyper_param_logs'