import numpy as np
from matplotlib import pyplot as plt

#InputData = np.array([[0, 0],
#                      [0, 1],
#                      [1, 0],
#                      [1, 1]])

#TargetData = np.array([[0, 1],
#                       [1, 0],
#                       [1, 0],
#                       [0, 1]])

InputData = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1],

                      [0, 1, 0, 0],
                      [0, 1, 0, 1],
                      [0, 1, 1, 0],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1],
                      [1, 0, 1, 0],
                      [1, 0, 1, 1],
                      [1, 1, 0, 0],
                      [1, 1, 0, 1],
                      
                      [1, 1, 1, 1]])

TargetData = np.array([[1, 0], 
                       [0, 1], 
                       [1, 0], 
                       [0, 1], 
                       [0, 1], 
                       [1, 0], 
                       [1, 0], 
                       [0, 1], 
                       [0, 1], 
                       [1, 0], 
                       [1, 0], 
                       [0, 1], 
                       [1, 0]])

def tanh_p(x):
    return 1 - np.tanh(x)**2

w1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)

w2 = np.random.randn(3, 3)
b2 = np.random.randn(3, 1)

w3 = np.random.randn(2, 3)
b3 = np.random.randn(2, 1)

iterations = 3000

lr = 0.2
costlist = []

for i in range(iterations):

    random = np.random.choice(len(InputData))

    z1 = np.dot(w1, InputData[random].reshape(4, 1)) + b1
    a1 = np.tanh(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)

    z3 = np.dot(w3, a2) + b3
    a3 = np.tanh(z3)
    
    cost = np.sum(np.square(a3 - TargetData[random].reshape(2, 1)))

    #print(a3)
    #print(cost)

    if i % 20 == 0:
        c = 0
        for x in range(len(InputData)):

            z1 = np.dot(w1, InputData[x].reshape(4, 1)) + b1
            a1 = np.tanh(z1)

            z2 = np.dot(w2, a1) + b2
            a2 = np.tanh(z2)

            z3 = np.dot(w3, a2) + b3
            a3 = np.tanh(z3)

            c += np.sum(np.square(a3 - TargetData[x].reshape(2, 1)))
        costlist.append(float(c))

    #backprop

    dcda3 = 2 * (a3 - TargetData[random].reshape(2, 1))
    da3dz3 = tanh_p(z3)
    dz3dw3 = a2

    dz3da2 = w3
    da2dz2 = tanh_p(z2)
    dz2dw2 = a1

    dz2da1 = w2
    da1dz1 = tanh_p(z1)
    dz1dw1 = InputData[random].reshape(4, 1)

    dw3 = dcda3 * da3dz3
    db3 = np.sum(dw3, axis=1, keepdims=True)
    w3 = w3 - lr * np.dot(dw3, dz3dw3.T)
    b3 = b3 - lr * db3

    dw2 = np.dot(dz3da2.T, dw3) * da2dz2
    db2 = np.sum(dw2, axis=1, keepdims=True)
    w2 = w2 - lr * np.dot(dw2, dz2dw2.T)
    b2 = b2 - lr * db2

    dw1 = np.dot(dz2da1.T, dw2) * da1dz1
    db1 = np.sum(dw1, axis=1, keepdims=True)
    w1 = w1 - lr * np.dot(dw1, dz1dw1.T)
    b1 = b1 - lr * db1

z1 = np.dot(w1, InputData[5].reshape(4, 1)) + b1
a1 = np.tanh(z1)

z2 = np.dot(w2, a1) + b2
a2 = np.tanh(z2)

z3 = np.dot(w3, a2) + b3
a3 = np.tanh(z3)

cost = np.sum(np.square(a3 - TargetData[5].reshape(2, 1)))

print("Prediction: \n{}\n".format(np.round(a3)))
print("Target: \n{}\n".format(TargetData[5].reshape(2, 1)))
print("Cost: {}".format(cost))

plt.plot(costlist)
plt.show()
