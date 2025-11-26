import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

test_loss = np.load('test_loss.npy')
train_loss = np.load('training_loss.npy')

plt.figure(figsize=(8, 6))
plt.plot(test_loss, label='test loss')
plt.plot(train_loss, label='train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()