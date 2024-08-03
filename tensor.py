import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,shutil
import cv2
import matplotlib.image as mpimg
import seaborn as sns
plt.style.use('ggplot')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import warnings

import zipfile
z = zipfile.ZipFile('File.zip')
z.extractall()

folder = 'brain_tumor_dataset/yes/'
count =1
for i in os.listdir(folder):
    source = folder + i
    destination = folder + "Y_" +str(count)+".jpg"
    os.rename(source,destination)
    count+=1

print("All files are renamed in the yes directory.")

folder = 'brain_tumor_dataset/no/'
count =1
for i in os.listdir(folder):
    source = folder + i
    destination = folder + "N_" +str(count)+".jpg"
    os.rename(source,destination)
    count+=1

print("All files are renamed in the no directory.")

listyes = os.listdir("brain_tumor_dataset/yes/")
number_files_yes = len(listyes)
print(number_files_yes)

listno = os.listdir("brain_tumor_dataset/no/")
number_files_no = len(listno)
print(number_files_no)

data = {'tumorous':number_files_yes,'non-tumorous':number_files_no}
types = data.keys()
values = data.values()

fig = plt.figure(figsize=(5,5))
plt.bar(types,values,color="blue")
plt.xlabel("Data")
plt.ylabel("No. of Brain MRI images")
plt.title("Count of Brain tumour images")
plt.show()

def timing(sec_elapsed):
    h = int(sec_elapsed/(60*60))
    m = int(sec_elapsed%(60*60)/60)
    s = sec_elapsed%60

    return f"{h}:{m}:{s}"

def augmented_data(file_dir,n_generated_sample,save_to_dir):
    data_gen = ImageDataGenerator(rotation_range=10,
                      width_shift_range =0.1,
                      height_shift_range = 0.1,
                      shear_range=0.1,
                      brightness_range=(0.3,1.0),
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='nearest')
    for filename in os.listdir(file_dir):
        image = cv2.imread(file_dir+'/'+filename)
        image = image.reshape((1,)+image.shape)
        save_prefix = 'aug_'+filename[:-4]
        i=0
        for batch in data_gen.flow(x=image,batch_size=1,save_to_dir=save_to_dir,save_prefix = save_prefix,save_format="jpg"):
            i+=1
            if i>n_generated_sample:
                break
            
            
import time
start_time = time.time()

yes_path= 'brain_tumor_dataset/yes'
no_path='brain_tumor_dataset/no'

augmented_data_path = 'augmented_data/'
augmented_data(file_dir = yes_path , n_generated_sample= 6 , save_to_dir=augmented_data_path+'yes')
augmented_data(file_dir = no_path , n_generated_sample = 9 , save_to_dir=augmented_data_path+'no')

end_time = time.time()

execution_time = end_time - start_time
print(timing(execution_time))

def data_summary(main_data):
    yes_path = "augmented_data/yes/"
    no_path = "augmented_data/no/"
    
    n_pos = len(os.listdir(yes_path))
    n_neg = len(os.listdir(no_path))
    
    n=n_pos+n_neg
    pos_per=(n_pos*100)/n
    neg_per=(n_neg*100)/n
    print(f"Number of samples : {n}")
    print(f"{n_pos} Number of positive sample in percentage : {pos_per}%")
    print(f"{n_neg} Number of negative sample in percentage : {neg_per}%")
    
data_summary('augmented_data/')

listyes = os.listdir("augmented_data/yes/")
number_files_yes = len(listyes)
print(number_files_yes)

listno = os.listdir("augmented_data/no/")
number_files_no = len(listno)
print(number_files_no)

data = {'tumorous':number_files_yes,'non-tumorous':number_files_no}
types = data.keys()
values = data.values()

fig = plt.figure(figsize=(5,5))
plt.bar(types,values,color="blue")
plt.xlabel("Data")
plt.ylabel("No. of Brain MRI images")
plt.title("Count of Brain tumour images")
plt.show()


import imutils
def crop_brain_tumor(image,plot=False):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    
    thres=cv2.threshold(gray,45,255,cv2.THRESH_BINARY)[1]
    thres=cv2.erode(gray,None,iterations=2)
    thres=cv2.dilate(thres,None,iterations=2)
    
    cnts=cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    c = max(cnts,key=cv2.contourArea)
    
    extLeft = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    extTop = tuple(c[c[:,:,1].argmin()][0])
    extBottom = tuple(c[c[:,:,1].argmax()][0])
    
    new_image = image[extTop[1]:extBottom[1],extLeft[0]:extRight[0]]
    if plot:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.tick_params(axis='both',which='both',top=False,bottom=False,left=False,right=False,
                        labelbottom=False,labeltop=False,labelleft=False,labelright=True)
        plt.title("Original Image")
        
        plt.figure()
        plt.subplot(1,2,2)
        plt.imshow(new_image)
        plt.tick_params(axis='both',which='both',top=False,bottom=False,left=False,right=False,
                        labelbottom=False,labeltop=False,labelleft=False,labelright=True)
        plt.title("Cropped Image")
        plt.show()
    
    return new_image

img = cv2.imread('augmented_data/yes/aug_Y_1_0_4360.jpg')
crop_brain_tumor(img,True)



from sklearn.utils import shuffle
def load_data(dir_list,image_size):
    X=[]
    y=[]
    image_width,image_height=image_size
    
    for directory in dir_list:
        for filename in os.listdir(directory):
            image=cv2.imread(directory+'/'+filename)
            image=crop_brain_tumor(image,plot=False)
            image=cv2.resize(image,dsize=(image_width,image_height),interpolation=cv2.INTER_CUBIC)
            image=image/255.00
            X.append(image)
            if directory[-3:]=="yes":
                y.append(1)
            else:
                y.append(0)
    
    X=np.array(X)
    y=np.array(y)
    X,y=shuffle(X,y)
    print(f"number of example is : {len(X)}")
    print(f"X shape is : {X.shape}")
    print(f"y shape is : {y.shape}")
    return X,y


augmented_path  = 'augmented_data/'
augmented_yes = augmented_path + 'yes'
augmented_no = augmented_path + 'no'
IMAGE_WIDTH,IMAGE_HEIGHT = (240,240)
load_data([augmented_yes,augmented_no],(IMAGE_WIDTH,IMAGE_HEIGHT))    

def plot_sample_images(X,y,n=50):
    for label in [0,1]:
        images = X[np.argwhere(y==label)]
        n_images = images[:n]
        
        column_n = 10
        rows_n = int(n/column_n)
        plt.figure(figsize=(20,10))
        i=1
        for image in n_images:
            plt.subplot(rows_n,column_n,i)
            plt.imshow(image[0])
            plt.tick_params(axis='both',which='both',
                            top=False,bottom=False,left=False,right=False,
                            labelbottom=False,labeltop=False,labelleft=False,labelright=False)
            i+=1
            
        label_to_str = lambda label: "Yes" if label ==1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()
                
plot_sample_images(X,y)

if not os.path.isdir('tumorous_and_nontumorous'):
    base_dir = 'tumorous_and_nontumorous'
    os.mkdir(base_dir)
    
base_dir = 'tumorous_and_nontumorous'

if not os.path.isdir('tumorous_and_nontumorous/train'):
    train_dir = os.path.join(base_dir,'train')
    os.mkdir(train_dir)
    
if not os.path.isdir('tumorous_and_nontumorous/test'):
    test_dir = os.path.join(base_dir,'test')
    os.mkdir(test_dir)
    
if not os.path.isdir('tumorous_and_nontumorous/validation'):
    validation_dir = os.path.join(base_dir,'validation')
    os.mkdir(validation_dir)

train_dir = 'tumorous_and_nontumorous/train'
test_dir='tumorous_and_nontumorous/test'
validation_dir='tumorous_and_nontumorous/validation'

infected_train = os.path.join(train_dir,'tumorous')
os.mkdir(infected_train)
infected_test = os.path.join(test_dir,'tumorous')
os.mkdir(infected_test)
infected_validation = os.path.join(validation_dir,'tumorous')
os.mkdir(infected_validation)

healthy_train = os.path.join(train_dir,'nontumorous')
os.mkdir(healthy_train)
healthy_test = os.path.join(test_dir,'nontumorous')
os.mkdir(healthy_test)
healthy_validation = os.path.join(validation_dir,'nontumorous')
os.mkdir(healthy_validation)

infected_train = 'tumorous_and_nontumorous/train/tumorous'
healthy_train = 'tumorous_and_nontumorous/train/nontumorous'

infected_test = 'tumorous_and_nontumorous/test/tumorous'
healthy_test = 'tumorous_and_nontumorous/test/nontumorous'

infected_validation = 'tumorous_and_nontumorous/validation/tumorous'
healthy_validation = 'tumorous_and_nontumorous/validation/nontumorous'

original_dataset_tumorous = os.path.join('augmented_data','yes/')
original_dataset_nontumorous = os.path.join('augmented_data','no/')

files = os.listdir('augmented_data/yes/')
fnames=[]
for i in range(0,1700):
    fnames.append(files[i])
for fname in fnames:
    src=os.path.join(original_dataset_tumorous,fname)
    dst=os.path.join(infected_train,fname)
    shutil.copy(src,dst)
    

files = os.listdir('augmented_data/yes/')
fnames=[]
for i in range(1700,1950):
    fnames.append(files[i])
for fname in fnames:
    src=os.path.join(original_dataset_tumorous,fname)
    dst=os.path.join(infected_test,fname)
    shutil.copy(src,dst)

files = os.listdir('augmented_data/yes/')
fnames=[]
for i in range(1950,2170):
    fnames.append(files[i])
for fname in fnames:
    src=os.path.join(original_dataset_tumorous,fname)
    dst=os.path.join(infected_validation,fname)
    shutil.copy(src,dst)

files = os.listdir('augmented_data/no/')
fnames=[]
for i in range(0,1400):
    fnames.append(files[i])
for fname in fnames:
    src=os.path.join(original_dataset_nontumorous,fname)
    dst=os.path.join(healthy_train,fname)
    shutil.copy(src,dst)
    
files = os.listdir('augmented_data/no/')
fnames=[]
for i in range(1400,1700):
    fnames.append(files[i])
for fname in fnames:
    src=os.path.join(original_dataset_nontumorous,fname)
    dst=os.path.join(healthy_test,fname)
    shutil.copy(src,dst)
    
files = os.listdir('augmented_data/no/')
fnames=[]
for i in range(1700,1959):
    fnames.append(files[i])
for fname in fnames:
    src=os.path.join(original_dataset_nontumorous,fname)
    dst=os.path.join(healthy_validation,fname)
    shutil.copy(src,dst)

train_datagen = ImageDataGenerator(rescale = 1.0/255,
                   horizontal_flip = 0.4,
                   vertical_flip = 0.4,
                   rotation_range = 40,
                   shear_range = 0.2,
                   width_shift_range = 0.4,
                   height_shift_range = 0.4,
                   fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1.0/255)
validation_datagen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory('tumorous_and_nontumorous/train/',batch_size = 32,target_size=(240,240),class_mode = 'categorical',
                                  shuffle = True,seed=42,color_mode ='rgb')

test_generator = train_datagen.flow_from_directory('tumorous_and_nontumorous/test/',batch_size = 32,target_size=(240,240),class_mode = 'categorical',
                                  shuffle = True,seed=42,color_mode ='rgb')

validation_generator = train_datagen.flow_from_directory('tumorous_and_nontumorous/validation/',batch_size = 32,target_size=(240,240),class_mode = 'categorical',
                                  shuffle = True,seed=42,color_mode ='rgb')

class_labels = train_generator.class_indices
class_name = {value:key for(key,value) in class_labels.items()}
# print(class_name)

base_model = VGG19(input_shape = (240,240,3),include_top=False, weights='imagenet')
for layers in base_model.layers:
    layers.trainable=False
x=base_model.output
flat=Flatten()(x)

class_1 = Dense(4608,activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152,activation='relu')(drop_out)
output = Dense(2,activation = 'softmax')(class_2)

model_01 = Model(base_model.input,output)
# model_01.summary()

filepath='model.keras'
es = EarlyStopping(monitor='val_loss',verbose=1,mode='min',patience=4)
cp = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False,
                mode='auto',save_freq='epoch')
lrr = ReduceLROnPlateau(monitor='validation_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.0001)

sgd = SGD(learning_rate=0.0001,decay=1e-6,momentum=0.9,nesterov=True)
model_01.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
history_01 = model_01.fit(train_generator,steps_per_epoch=10,epochs=20,callbacks=[es,cp,lrr],validation_data=validation_generator)

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
fig.suptitle("Model Training (Frozen CNN)",fontsize=12)
max_epoch=len(history_01.history['accuracy'])+1
epochs_list = list(range(1,max_epoch))

ax1.plot(epochs_list,history_01.history['accuracy'],color='b',linestyle='-',label='Training Data')
ax1.plot(epochs_list,history_01.history['val_loss'],color='r',linestyle='-',label='Validation Data')
ax1.set_title('Training Accuracy',fontsize=12)
ax1.set_xlabel('Epochs',fontsize=12)
ax1.set_ylabel('Accuracy',fontsize=12)
ax1.legend(frameon=False,loc='lower center',ncol=2)

ax2.plot(epochs_list,history_01.history['loss'],color='b',linestyle='-',label='Training Data')
ax2.plot(epochs_list,history_01.history['val_loss'],color='r',linestyle='-',label='Validation Data')
ax2.set_title('Training Loss',fontsize=12)
ax2.set_xlabel('Epochs',fontsize=12)
ax2.set_ylabel('Loss',fontsize=12)
ax2.legend(frameon=False,loc='upper center',ncol=2)
plt.savefig("training_frozencnn.jpeg",format='jpeg',dpi=100,bbox_inches='tight')