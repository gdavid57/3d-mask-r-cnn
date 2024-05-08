import os
import argparse

from core.config import load_config
from core.models import RPN

# Comment the following line to debug TF or libcuda issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/rpn/embryo_rpn_config.json')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--summary', type=bool, default=True)
    args = parser.parse_args()

    # Load training config
    toy_config = load_config(args.config_path)

    # Assign gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Initiate model
    rpn = RPN(toy_config, show_summary=args.summary)
    
    # Training loop (one epoch -> save weights -> quick train monitoring -> one epoch -> ...)
    rpn.train()
