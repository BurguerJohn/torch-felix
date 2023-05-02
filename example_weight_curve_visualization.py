import numpy as np
import matplotlib.pyplot as plt
from felix import CreateWeights

#Weights have a float for the start and the end of the model, it will interpolate between both values as the depth get closer to the end of the model.

print("Linear 1-1")
data = CreateWeights(1.0, 1.0, 10, 1)
plt.title("Linear 1-1")
plt.xlabel('Depth')
plt.ylabel('Weight')
plt.plot(data)
plt.savefig('img/Linear 1-1.png')
plt.show()

print("Linear 0-1")
data = CreateWeights(0., 1.0, 10, 1)
plt.title("Linear 0-1")
plt.xlabel('Depth')
plt.ylabel('Weight')
plt.plot(data)
plt.savefig('img/Linear 0-1.png')
plt.show()

print("Curve 0-1")
data = CreateWeights(0., 1.0, 10, 2)
plt.title("Curve 0-1")
plt.xlabel('Depth')
plt.ylabel('Weight')
plt.plot(data)
plt.savefig('img/Curve 0-1.png')
plt.show()

print("Curve Strong 0-1")
data = CreateWeights(0., 1.0, 10, 4)
plt.title("Curve Strong 0-1")
plt.xlabel('Depth')
plt.ylabel('Weight')
plt.plot(data)
plt.savefig('img/Curve Strong 0-1.png')
plt.show()

print("Curve 1 - 0.01")
data = CreateWeights(1.0, 0.01, 10, 2)
plt.title("Curve 1 - 0.01")
plt.xlabel('Depth')
plt.ylabel('Weight')
plt.plot(data)
plt.savefig('img/Curve 1 - 0.01.png')
plt.show()