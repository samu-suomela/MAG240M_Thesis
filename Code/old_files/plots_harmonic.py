import matplotlib.pyplot as plt
import numpy as np

Dataset = ["Small","Medium","Large","XL","XXL"]
Accuracy = [0.383,0.296,0.218,0.294,0.350]

plt.bar(Dataset, Accuracy)
plt.yticks(np.arange(0, 0.6, step=0.2))
plt.title('Prediction accuracy')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.show()

Time = [0.646, 0.293, 2.992, 21.495, 115.726]

plt.plot(Dataset, Time)
plt.yticks(np.arange(0, 120, step=10))
plt.title('Computational speed')
plt.xlabel('Dataset')
plt.ylabel('Time (s)')
plt.show()

