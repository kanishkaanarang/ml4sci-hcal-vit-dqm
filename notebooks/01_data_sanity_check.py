import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("data/Run355456_Dataset.npy")
data2 = np.load("data/Run357479_Dataset.npy")

print("Run355456 shape:", data1.shape)
print("Run357479 shape:", data2.shape)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(data1[0], aspect="auto")
plt.title("Run355456 – LS 0")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(data2[0], aspect="auto")
plt.title("Run357479 – LS 0")
plt.colorbar()

plt.tight_layout()
plt.show()
