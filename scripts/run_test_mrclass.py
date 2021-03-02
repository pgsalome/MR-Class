
#import sys
#import os
#sys.path.append(os.getcwd())

from mrclass_resnet.utils import load_config
from mrclass_resnet.test_mrclass import test
import argparse

if __name__ == "__main__":
   
    print('#'*20)
    print("MR-Class")
    print('#'*20)
    parser = argparse.ArgumentParser(description='MR-Class')
    parser.add_argument('--config', type = str, help = 'Path to the configuration file', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    test(config)

    
    print('Done')

