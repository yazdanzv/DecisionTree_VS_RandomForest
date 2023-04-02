import pandas as pd

PATH = ".\\dataset_simple.csv"

def read_x():
    li = [range(55)]
    df = pd.read_csv(PATH, usecols=[])