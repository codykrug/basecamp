import click

with click.progressbar(range(1000000)) as bar:
    for i in bar:
        pass


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(\
num_words=10000)


#exploring the data

print("Training entries: {}, labels {}".format(\
len(train_data), len(train_labels)))

print(train_data[0])
len(train_data[0]), len(train_data[1])



# A dictionary mapping words into an integer index
word_index = imdb.get_word_index()


#the first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#picking a random review to view
decode_review(train_data[12])



#preparing the data, the need to embed or pad

train_data = keras.preprocessing.sequence.pad_sequences(\
  train_data,
  value=word_index["<PAD>"],
  padding="post",
  maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(\
test_data,
value=word_index["<PAD>"],
padding="post",
maxlen=256)


#again look at the length now

len(train_data[12]), len(train_data[13])
print(train_data[0])


#now we build the model

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 160))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(160, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()



#loss function

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


#creating a cross validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#train the model

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


#evaluate the model and graph it

results = model.evaluate(test_data, test_labels)

print(results)



history_dict = history.history
history_dict.keys()


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.pdf')

#need to clear figure

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.pdf')                                          
