from scripts.logger import Logger
from scripts.utils import setup_data_loaders, setup_model, num_parameters

#
# Utility script that has been used to find the hyperparameters (e.g., layer width and number of squares)
# of the circuits for the UCI data sets, such that they have approximately the same number of
# learnable parameters
#

if __name__ == '__main__':
    seed = 42
    logger = Logger('progressive-size', verbose=True)
    metadata, _ = setup_data_loaders('power', 'datasets', logger)

    for conf in [(1, 152, 256), (4, 102, 64), (8, 80, 32), (16, 60, 16)]:
        num_components, num_units, num_input_units = conf
        model = setup_model(
            'SOS',
            metadata,
            logger=logger,
            region_graph='rnd-bt',
            structured_decomposable=True,
            num_components=num_components,
            num_units=num_units,
            num_input_units=num_input_units,
            complex=True,
            seed=seed
        )
        print(f"Conf: {conf}")
        num_params = num_parameters(model)
        num_sum_params = num_parameters(model, sum_only=True)
        num_input_params = num_params - num_sum_params
        print(f"Overall parameters: {num_params}")
        print(f"Sum-only parameters: {num_sum_params}")
        print(f"Input-only parameters: {num_input_params}")
