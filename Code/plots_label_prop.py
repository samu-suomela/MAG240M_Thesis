import matplotlib.pyplot as plt
import numpy as np

Dataset = ["Small","Medium","Large","XL","XXL"]
Accuracy = [0.362,0.390,0.647,0,0]

plt.bar(Dataset, Accuracy)
plt.yticks(np.arange(0, 1.2, step=0.2))
plt.title('Prediction accuracy')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.show()

Time = [0.13, 6.25, 196.93, 0, 0]

plt.plot(Dataset, Time)
plt.yticks(np.arange(0, 1500, step=100))
plt.title('Computational speed')
plt.xlabel('Dataset')
plt.ylabel('Time (min)')
plt.show()

