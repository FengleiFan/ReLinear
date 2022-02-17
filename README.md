# Expressivity and Trainability of Quadratic Networks
| [ArXiv](https://arxiv.org/abs/2201.05279) |

This respository includes implementations of the algorithm "ReLInear" proposed in *Expressivity and Trainability of Quadratic Networks*.  The ReLinear encourages the model to learn suitable quadratic terms gradually and adaptively in reference to the corresponding linear terms. The ReLinear method has the following two steps. First, the quadratic weights in each neuron are set to $\textbf{w}^g = 0, b^g = 1$ and $\textbf{w}^b = 0, c = 0$. Such an initialization degenerates a quadratic neuron into a conventional neuron. Second, quadratic terms are regularized in the training process. Intuitively, two ways of regularization: shrinking the gradients of quadratic weights (ReLinear$^{sg}$); and shrinking quadratic weights (ReLinear$^{sw}$). 

<p align="center">
  <img width="320" src="https://github.com/FengleiFan/ReLinear/blob/main/Figure_guaranteed_improvements.png">
</p>

<p align="center">
  Figure 1. The performance of a quadratic network trained using the proposed \textit{ReLinear} method, with an observed improvement than the conventional network of the same structure. $(\gamma_g,\gamma_b)$, $(\alpha_g,\alpha_b)$, and $(\beta_g,\beta_b)$ are hyperparameters of \textit{ReLinear}. As these hyperparameters increases from $0$, the trained model transits from the conventional model to the quadratic, and the model's performance reaches the optimality.
</p>
