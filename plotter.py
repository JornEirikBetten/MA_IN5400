import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import pandas as pd
import numpy as np
import os
dir = os.getcwd()
measures = pd.read_csv(dir + '/losses.csv')
avg_p_scores = pd.read_csv(dir + '/classes.csv')


list = []

num_arr = np.zeros((2, 2))
num_arr2 = np.ones((2, 2))
list.append(num_arr)
list.append(num_arr2)
print(list[0])

epochs = np.linspace(1, 20, 20)

plt.figure()
plt.plot(epochs, measures['training_loss'], label='Training loss')
plt.plot(epochs, measures['validation_loss'], label="Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
           'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
           'blow_down', 'conventional_mine', 'cultivation', 'habitation',
           'primary', 'road', 'selective_logging', 'slash_burn', 'water']
plt.figure()
i = 1
for header in classes:
    plt.plot(epochs, avg_p_scores[header])
    i += 1

plt.xlabel('Epoch')
plt.ylabel('Average precision score')
plt.show()

plt.figure()
plt.plot(epochs, measures['mean_average_score'], label="mAP")
plt.xlabel('Epoch')
plt.xticks(ticks=[5, 10, 15, 20], labels=["5", "10", "15", "20"])
plt.ylabel('Mean average precision')
plt.legend()
plt.show()
