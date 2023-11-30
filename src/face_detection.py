import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict
# visualization
from PIL import Image
# identifying faces
from mtcnn.mtcnn import MTCNN
# visualizing bounding boxes
import matplotlib.patches as patches
# CNN
import keras
from sklearn.model_selection import train_test_split
# Moving files between directories
import shutil
from shutil import unpack_archive
from subprocess import check_output

DATA_PATH = '../data/raw/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled'


lfw_allnames = pd.read_csv("../data/lfw-dataset/lfw_allnames.csv")
matchpairsDevTest = pd.read_csv("../data/lfw-dataset/matchpairsDevTest.csv")
matchpairsDevTrain = pd.read_csv("../data/lfw-dataset/matchpairsDevTrain.csv")
mismatchpairsDevTest = pd.read_csv("../data/lfw-dataset/mismatchpairsDevTest.csv")
mismatchpairsDevTrain = pd.read_csv("../data/lfw-dataset/mismatchpairsDevTrain.csv")
pairs = pd.read_csv("../data/lfw-dataset/pairs.csv")
# tidy pairs data: 
pairs = pairs.rename(columns ={'name': 'name1', 'Unnamed: 3': 'name2'})
matched_pairs = pairs[pairs["name2"].isnull()].drop("name2",axis=1)
mismatched_pairs = pairs[pairs["name2"].notnull()]
people = pd.read_csv("../data/lfw-dataset/people.csv")
# remove null values
people = people[people.name.notnull()]
peopleDevTest = pd.read_csv("../data/lfw-dataset/peopleDevTest.csv")
peopleDevTrain = pd.read_csv("../data/lfw-dataset/peopleDevTrain.csv")


print("hello")