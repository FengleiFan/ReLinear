# Expressivity and Trainability of Quadratic Networks
| [ArXiv](https://arxiv.org/abs/2201.05279) |

This respository includes implementations of the algorithm "ReLInear" proposed in *Expressivity and Trainability of Quadratic Networks*.  The ReLinear encourages the model to learn suitable quadratic terms gradually and adaptively in reference to the corresponding linear terms. The ReLinear method has the following two steps. First, the quadratic weights in each neuron are set to $w^g = 0, b^g = 1$ and $w^b = 0, c = 0$. Such an initialization degenerates a quadratic neuron into a conventional neuron. Second, quadratic terms are regularized in the training process. Intuitively, two ways of regularization: shrinking the gradients of quadratic weights and shrinking quadratic weights. 

<p align="center">
  <img width="320" src="https://github.com/FengleiFan/ReLinear/blob/main/Figure_IWL.png">
</p>

<p align="center">
  Figure 1. Illustration of the proposed training strategy.
</p>

<p align="center">
  <img width="320" src="https://github.com/FengleiFan/ReLinear/blob/main/Figure_guaranteed_improvements.png">
</p>

<p align="center">
  Figure 2. The performance of a quadratic network trained using the proposed ReLinear method, with an observed improvement than the conventional network of the same structure. $(\gamma_g,\gamma_b)$, $(\alpha_g,\alpha_b)$, and $(\beta_g,\beta_b)$ are hyperparameters of ReLinear. As these hyperparameters increases from 0, the trained model transits from the conventional model to the quadratic, and the model's performance reaches the optimality.
</p>



