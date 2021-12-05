import matplotlib.pyplot as plt
import numpy as np

Dataset = ["Small","Medium","Large","XL","XXL"]
Accuracy = [0.02128,0.03774,0.02802,0.03074,0.02151]

plt.bar(Dataset, Accuracy)
plt.yticks(np.arange(0, 0.04, step=0.01))
plt.title('Prediction accuracy')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.show()

Time = [0.14, 1.1, 7.0, 18.4, 54.8]

plt.plot(Dataset, Time)
plt.yticks(np.arange(0, 70, step=10))
plt.title('Computational speed')
plt.xlabel('Dataset')
plt.ylabel('Time (min)')
plt.show()

