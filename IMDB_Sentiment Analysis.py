#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.import packages
import csv
import numpy as np
import pickle
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, classification_report, f1_score
from itertools import chain

from keras.utils.data_utils import get_file
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding,Dense, Dropout, Activation,Flatten
from keras.layers import LSTM,SimpleRNN
from keras.datasets import imdb


# In[2]:


# Get full dataset from the webside
path = get_file('imdb_full.pkl',
                origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl')

# Split into train/test, and separate features from labels
f = open(path, 'rb')
(x_train, labels_train), (x_test, labels_test) = pickle.load(f)



# Using the Index/word mapping in keras.datasets to get the csv of the train and test dataset

#train dataset
idx = imdb.get_word_index()
idx2word = {v: k for k, v in idx.items()}
with open('train.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    for i in range(0, len(x_train)):
        label = labels_train[i]
        review = ' '.join([idx2word[o] for o in x_train[i]])
        writer.writerow([review, label])

#test dataset
with open('test.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    for i in range(0, len(x_test)):
        label = labels_test[i]
        review = ' '.join([idx2word[o] for o in x_test[i]])
        writer.writerow([review, label])


# In[3]:


#ouput the train dataset
train_data = pd.read_csv('train.csv', header=None)
print(train_data.shape)
train_data


# In[4]:


#ouput the train dataset
test_data = pd.read_csv('test.csv', header=None)
print(test_data.shape)
test_data


# ### The train and test dataset provide a set of 25,000 highly polar movie reviews with tag 1: positive reviews; tag 0: negative reviews

# In[11]:


#load the dataset and save the top 5000 words 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=7000)


# # Model1:Multi-Layer Perceptron(MLP)

# In[17]:


#Model 3: MLP

#data preprocessing
# Pad sequences: Convert the sequence into a new sequence of the same length with 400.
x_train = sequence.pad_sequences(x_train, maxlen=400)
x_test = sequence.pad_sequences(x_test, maxlen=400)
print('Train data size:', x_train.shape)
print('Test data size:', x_test.shape)

#set parameters 
max_features = 7000
embedding_size = 50
maxlen = 400


batch_size = 64
epochs = 4


print('Build the MLP model...')
model1 = Sequential()
model1.add(Embedding(max_features, 
                    embedding_size, 
                    input_length=maxlen))
model1.add(Dropout(0.35))

model1.add(Flatten())

model1.add(Dense(units=256, activation='relu'))
model1.add(Dropout(0.35))

model1.add(Dense(units=1,activation='sigmoid'))
           
model1.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model1.summary()


# In[18]:


hist1=model1.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# In[29]:


# Evaluate MLP model

acc=hist1.history['accuracy']
val_acc=hist1.history['val_accuracy']
loss=hist1.history['loss']
val_loss=hist1.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs,acc,'b',label='Training acc')
plt.plot(epochs,val_acc,'r',label='Validation acc')
plt.title("Training and validation accuracy for MLP")
plt.legend()
plt.figure()

score, acc = model1.evaluate(x_test, y_test, batch_size=batch_size)
preds = model1.predict_classes(x_test, batch_size=batch_size)

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
def plot_confusion_matrix(cm, classes, normalize=False, title='LSTM Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.ylim(-0.5,1.5)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],verticalalignment='center', horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#F1-Score
f1_macro = f1_score(y_test, preds, average='macro') 
f1_micro = f1_score(y_test, preds, average='micro')

print('Test accuracy:', acc)
print('Test score (loss):', score)
print('')
print('F1 Score (Macro):', f1_macro)
print('F1 Score (Micro):', f1_micro)

plot_confusion_matrix(cm, {'negative': 0, 'positive': 1})


# # Model2: Recurrent  Neural Network(RNN)

# In[20]:


#Model 2: RNN

#data preprocessing
# Pad sequences: Convert the sequence into a new sequence of the same length with 400.
x_train = sequence.pad_sequences(x_train, maxlen=400)
x_test = sequence.pad_sequences(x_test, maxlen=400)
print('Train data size:', x_train.shape)
print('Test data size:', x_test.shape)

#set parameters 
max_features = 7000
embedding_size = 50
maxlen = 400

batch_size = 64
epochs = 4



print('Build the RNN model...')
model2 = Sequential()

# Embedding layer
model2.add(Embedding(max_features, 
                    embedding_size, 
                    input_length=maxlen))
model2.add(Dropout(0.35))

model2.add(SimpleRNN(units=16))

model2.add(Dense(units=256,activation='relu'))
model2.add(Dropout(0.35))
model2.add(Dense(units=1,activation='sigmoid'))

model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.summary()


# In[21]:


hist2=model2.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         validation_data=(x_test, y_test),
         verbose=1)


# In[28]:


# Evaluate RNN model
acc=hist2.history['accuracy']
val_acc=hist2.history['val_accuracy']
loss=hist2.history['loss']
val_loss=hist2.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs,acc,'b',label='Training acc')
plt.plot(epochs,val_acc,'r',label='Validation acc')
plt.title("Training and validation accuracy for RNN")
plt.legend()
plt.figure()

score, acc = model2.evaluate(x_test, y_test, batch_size=batch_size)
preds = model2.predict_classes(x_test, batch_size=batch_size)

# Confusion Matrix
cm = confusion_matrix(y_test, preds)

#F1-Score
f1_macro = f1_score(y_test, preds, average='macro') 
f1_micro = f1_score(y_test, preds, average='micro')


print('Test accuracy:', acc)
print('Test score (loss):', score)
print('')
print('F1 Score (Macro):', f1_macro)
print('F1 Score (Micro):', f1_micro)

plot_confusion_matrix(cm, {'negative': 0, 'positive': 1})


# # Model3:Long Short-Term Memory(LSTM)

# In[23]:


#Model 3: LSTM

#data preprocessing
# Pad sequences: Convert the sequence into a new sequence of the same length with 400.
x_train = sequence.pad_sequences(x_train, maxlen=400)
x_test = sequence.pad_sequences(x_test, maxlen=400)
print('Train data size:', x_train.shape)
print('Test data size:', x_test.shape)

#set parameters 
max_features = 7000
embedding_size = 50
maxlen = 400

batch_size = 64
epochs = 4


print('Build the LSTM model...')
model3 = Sequential()
model3.add(Embedding(max_features, 
                    embedding_size, 
                    input_length=maxlen))
model3.add(Dropout(0.35))

model3.add(LSTM(32))

model3.add(Dense(units=256, activation='relu'))
model3.add(Dropout(0.35))

model3.add(Dense(units=1,activation='sigmoid'))
           
model3.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model3.summary()


# In[24]:


#train model
hist3=model3.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# In[30]:


# Evaluate LSTM model
acc=hist3.history['accuracy']
val_acc=hist3.history['val_accuracy']
loss=hist3.history['loss']
val_loss=hist3.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs,acc,'b',label='Training acc')
plt.plot(epochs,val_acc,'r',label='Validation acc')
plt.title("Training and validation accuracy for LSTM")
plt.legend()
plt.figure()

score, acc = model3.evaluate(x_test, y_test, batch_size=batch_size)
preds = model3.predict_classes(x_test, batch_size=batch_size)

# Confusion Matrix
cm = confusion_matrix(y_test, preds)

#F1-Score
f1_macro = f1_score(y_test, preds, average='macro') 
f1_micro = f1_score(y_test, preds, average='micro')


print('Test accuracy:', acc)
print('Test score (loss):', score)
print('')
print('F1 Score (Macro):', f1_macro)
print('F1 Score (Micro):', f1_micro)
print('')
print('The LSTM confusion matrix is:')
plot_confusion_matrix(cm, {'negative': 0, 'positive': 1})


# In[31]:


# Save the model weights
model1.save('MLP_model.h5')
model1.save_weights('MLP_weights.h5')

model2.save('RNN_model.h5')
model2.save_weights('RNN_weights.h5')

model3.save('LSTM_model.h5')
model3.save_weights('LSTM_weights.h5')


# In[ ]:





# In[ ]:




