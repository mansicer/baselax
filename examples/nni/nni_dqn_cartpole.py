import os
from nni.experiment import Experiment

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

experiment = Experiment('local')

experiment.config.trial_command = 'python dqn_cartpole.py'
experiment.config.trial_code_directory = '..'

search_space = {
    'learning_rate': {'_type': 'loguniform', '_value': [0.000001, 0.001]},
    'epsilon_steps': {'_type': 'qloguniform', '_value': [5, 50000, 1]},
    'buffer_size': {'_type': 'qloguniform', '_value': [500, 5000000, 1]},
}

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 10

experiment.run(8080)
