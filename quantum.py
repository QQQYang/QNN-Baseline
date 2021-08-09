#encoding=utf-8
"""
Benchmark of current quantum network for random label
-------------------------------------------
Author: xxx
Email: xxx@xxx.com
"""
import yaml
from easydict import EasyDict
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.append('.')

from engine.train import QuantumTrainer, TorchTrainer

def get_opt():
    """
    Get parameters passed by python script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default = 'config/qnn/qnn_GD_64_0_0-wine-parallel.yml')
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--parallel", action='store_true')
    parser.add_argument("--torch-trainer", action='store_true')
    parser.add_argument("--test", action='store_true')
    opt = parser.parse_args()
    return opt

def get_config(config_file):
    """
    Read config from config yaml
    """
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    return config

def run_process(param):
    config_file, index, is_test = param
    cfg = get_config(config_file)

    trainer = QuantumTrainer(cfg, index=index, name=config_file, is_test=is_test)
    trainer.train()

if __name__=='__main__':
    opt = get_opt()
    cfg = get_config(opt.config_file)

    if opt.parallel is False:
        if opt.torch_trainer:
            trainer = TorchTrainer(cfg, index=opt.index, name=opt.config_file, is_test=opt.test)
        else:
            trainer = QuantumTrainer(cfg, index=opt.index, name=opt.config_file, is_test=opt.test)
        if opt.test:
            trainer.evaluate()
        else:
            trainer.train()
    else:
        # multiprocessing
        # config_dir = '/'.join(opt.config_file.split('/')[:-1])
        # config_files = os.listdir(config_dir)
        # config_files = [os.path.join(config_dir, config_file) for config_file in config_files]
        inputs = [(opt.config_file, i, opt.test) for i in range(10)]

        with Pool(len(inputs)) as pool:
            pool.map(run_process, inputs)
    