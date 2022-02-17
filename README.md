# Expressivity and Trainability of Quadratic Networks
| [ArXiv](https://arxiv.org/abs/2201.05279) |

This respository includes implementations of the algorithm "ReLInear" proposed in *Expressivity and Trainability of Quadratic Networks*.  The ReLinear encourages the model to learn suitable quadratic terms gradually and adaptively in reference to the corresponding linear terms. The ReLinear method has the following two steps. First, the quadratic weights in each neuron are set to $\textbf{w}^g = 0, b^g = 1$ and $\textbf{w}^b = 0, c = 0$. Such an initialization degenerates a quadratic neuron into a conventional neuron. Second, quadratic terms are regularized in the training process. Intuitively, two ways of regularization: shrinking the gradients of quadratic weights (ReLinear$^{sg}$); and shrinking quadratic weights (ReLinear$^{sw}$).


