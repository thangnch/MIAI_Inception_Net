import scipy.io
import os
import shutil
import random

# Create label.csv
with open("data\\labels.csv","w") as f:
    mat = scipy.io.loadmat("car_devkit\\cars_meta.mat")
    print(mat)
    for anno in mat["class_names"][0]:
        f.write(anno[0] +"\n")


# Process test folder
train_dir = "data\\test"
mat = scipy.io.loadmat("car_devkit\\cars_test_annos_withlabels.mat")
for anno in mat["annotations"][0]:
    print(anno)

    if random.random()<0.8:
        train_dir = "data\\train"
    else:
        train_dir = "data\\test"
    # Create folder in train
    try:
        print("Make folder : ", anno[4][0][0])
        os.mkdir(train_dir + "\\" + str(anno[4][0][0]))
    except:
        pass

    # Copy file
    print("Copy file: ",anno[5][0])
    shutil.copy("CarDataSet\\cars_test\\" + str(anno[5][0]), train_dir + "\\" + str(anno[4][0][0]) + "\\" + str(anno[5][0]) )


# Process train folder
train_dir = "data\\train"
mat = scipy.io.loadmat("car_devkit\\cars_train_annos.mat")
for anno in mat["annotations"][0]:
    print(anno)
    # Create folder in train
    try:
        print("Make folder : ", anno[4][0][0])
        os.mkdir(train_dir + "\\" + str(anno[4][0][0]))
    except:
        pass

    # Copy file
    print("Copy file: ",anno[5][0])
    shutil.copy("CarDataSet\\cars_train\\" + str(anno[5][0]), train_dir + "\\" + str(anno[4][0][0]) + "\\" + str(anno[5][0]) )