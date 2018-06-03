#!/usr/bin/env python3

# used to count the training and testing images 

import os

train_count = sum([len(files) for r, d, files in os.walk("seedlings-data/train")])

test_count = sum([len(files) for r, d, files in os.walk("seedlings-data/test")])

print(f"There are {train_count} images for training and {test_count} images for testing.")