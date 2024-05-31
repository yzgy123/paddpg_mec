import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="mec", help="name of the scenario script")
    parser.add_argument("--time_steps", type=int, default=20, help="maximum episode length")
    parser.add_argument("--episodes", type=int, default=500, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-4, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon greedy")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(10000), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=1, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-time_steps", type=int, default=20, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    args = parser.parse_args()

    return args
