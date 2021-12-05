import matplotlib.pyplot as plt
import numpy as np

Dataset = ["Small","Medium","Large","XL","XXL"]
Accuracy_LPA = [0.362,0.390,0.647,0,0]
Accuracy_Harm = [0.383,0.296,0.218,0.294,0.350]
Accuracy_GCN = [0.02128,0.03774,0.02802,0.03074,0.02151]

fig = plt.figure()
ax = fig.add_subplot(111)
ind = np.arange(5)
width = 0.2

Acc_LPA = ax.bar(ind-width, Accuracy_LPA, width, color='r')
Acc_Harm = ax.bar(ind, Accuracy_Harm, width, color='b')
Acc_GCN = ax.bar(ind+width, Accuracy_GCN, width, color='g')
ax.set_xlabel('Dataset')
ax.set_ylabel('Accuracy')
ax.set_title('Node Classification Accuracy')
ax.set_xticks(ind + width / 3)
ax.set_xticklabels( ("Small","Medium","Large","XL","XXL") )
ax.legend((Acc_LPA[0], Acc_Harm[0], Acc_GCN[0]), ("Label Propagation","Harmonic Functions", "GCNN"))
plt.show()

Time_LPA = [0.13, 6.25, 196.93, None, None]
Time_Harm = [0.011, 0.005, 0.050, 0.358, 1.929]
Time_GCN = [0.14, 1.1, 7.0, 18.4, 54.8]

plt.plot(Dataset, Time_LPA, color = "r", label = "Label Propagation")
plt.plot(Dataset, Time_Harm, color = "b", label = "Harmonic Functions")
plt.plot(Dataset, Time_GCN, color = "g", label = "GCNN")
plt.yticks(np.arange(0, 280, step=40))
plt.legend(loc="upper left")
plt.title('Computational speed')
plt.xlabel('Dataset')
plt.ylabel('Time (min)')
plt.show()

