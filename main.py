import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# read dataset
train_df = pd.read_csv(r'./sign-language-mnist/sign_mnist_train.csv')
test_df = pd.read_csv(r'./sign-language-mnist/sign_mnist_test.csv')

# convert dataset into np array for tensorflow
train_data = np.array(train_df, dtype = 'float32')
test_data = np.array(test_df, dtype = 'float32')

# split dataset
x_train = train_data[:, 1:]
y_train = train_data[:, 0]

x_test = test_data[:, 1:]
y_test = test_data[:, 0]

x_train, x_validate, y_train, y_validate = train_test_split(
     x_train , y_train, test_size = 0.2, random_state = 12345)

img = x_train[50,:].reshape((28,28))
plt.imshow(img)
plt.show()
print("test")
