from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import tensorflow as tf
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import GRU, Linear, Dropout, Module
from torch.cuda import is_available
import numpy as np
from torch.autograd import set_detect_anomaly
from torch import device, zeros, ones, cat, randn, log, gather, norm, sigmoid, tanh
from torch import is_floating_point
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pandas as pd
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import time
import json
import pickle
from queue import PriorityQueue


# Define constant
# Default word tokens
# #
set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
USE_CUDA = is_available()
device = device("cuda" if USE_CUDA else "cpu")
