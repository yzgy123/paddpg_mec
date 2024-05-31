from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch


if __name__ == '__main__':
    np.random.seed(1)
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns=[]
        for i in range(100):
            returns.append(runner.evaluate(i))
        print((sum(returns)/len(returns))/20)
    else:
        runner.run()
