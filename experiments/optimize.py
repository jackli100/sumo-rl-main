import argparse
import re

from numpy import mat
from test_class import generate_result_folder, TrafficMatrix, Train, ShowResults

def str_to_float_list(s):
    return list(map(float, s.split(',')))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train and evaluate traffic model.')

    parser.add_argument('--proportion_of_saturations', type=str_to_float_list, default=[0.75, 0.75, 0.75, 0.75], help='Proportion of saturations for traffic matrix')
    parser.add_argument('--net_path', type=int, default=2, help='Path to the network file,1 means 17-stage, 2 means 2-stage, 3 means 4-stage')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total number of timesteps for training')
    parser.add_argument('--num_of_episodes', type=int, default=10, help='Number of episodes for training')
    
    args = parser.parse_args()
    if args.net_path == 1:
        net_path = "sumo_rl/nets/2way-single-intersection/single-intersection-2.net.xml"
    elif args.net_path == 2:
        net_path = "sumo_rl/nets/2way-single-intersection/single-intersection-3.net.xml"
    elif args.net_path == 3:
        net_path = "sumo_rl/nets/2way-single-intersection/single-intersection-4.net.xml"
    results_folder = generate_result_folder()

    # Generate the traffic matrix
    matrix = TrafficMatrix(results_folder, args.proportion_of_saturations)
    matrix.create_xml()
    route_path = matrix.output_file

    # Initialize the trainer
    trainer = Train(
        output_folder=results_folder,
        net_file=net_path,
        route_file=route_path,
        total_timesteps=args.total_timesteps,
        num_of_episodes=args.num_of_episodes,
    )

    # Train the model
    trainer.optimize_hyperparameters()
    
