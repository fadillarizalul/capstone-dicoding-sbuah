#cek versi tensorflow
import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import seaborn as sns

!wget --no-check-certificate \
https://github.com/fadillarizalul/capstone-dicoding-sbuah/raw/main/dataset/3-fruits.zip \
  -O fruit.zip
  
!unzip /content/fruit.zip

import os
 #lokasi direktori dasar
dir_base ='/content/3-fruits'
os.listdir(dir_base)
print(os.listdir(dir_base))

#hitung jumlah file per direktori
fresh_banana = len(os.listdir('/content/3-fruits/fresh banana'))
fresh_mango = len(os.listdir('/content/3-fruits/fresh mango')) 
fresh_orange = len(os.listdir('/content/3-fruits/fresh orange')) 
rotten_banana = len(os.listdir('/content/3-fruits/rotten banana'))
rotten_mango = len(os.listdir('/content/3-fruits/rotten mango'))
rotten_orange = len(os.listdir('/content/3-fruits/rotten orange'))

print("Amount of fresh banana images:", fresh_banana)
print("Amount of fresh mango images:", fresh_mango)
print("Amount of fresh orange images:", fresh_orange)
print("Amount of rotten banana images:", rotten_banana)
print("Amount of rotten mango images:", rotten_mango)
print("Amount of rotten orange images:", rotten_orange)
