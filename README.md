# Network for pressure

This repository implements a network based on the approach described by:

    [1] Tompson, J., Schlachter, K., Sprechmann, P., & Perlin, K. (2017). Accelerating Eulerian Fluid Simulation With Convolutional Networks. In Proceedings of the 34th International Conference on Machine Learning (pp. 3424–3433). PMLR. Retrieved from https://proceedings.mlr.press/v70/tompson17a.html

This network makes essentially use of a Convolution Neural Network with pooling, resizing layers etc, to learn how to solve, in a physics informed manner
$\nabla^2 p = b$ where $ b = \nabla u $.

This is done by estimating $p$ through a neural network enforcing the following loss function to minimize:
$$\mathcal{L} = \mathrm{MSE} (\nabla u, \nabla^2 p)$$
