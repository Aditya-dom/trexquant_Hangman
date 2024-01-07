import pandas as pd
import numpy as np
from utils import *
import train

if __name__ == '__main__':
    input_tensor, target_tensor = get_datasets()
    train_model(input_tensor, target_tensor)
