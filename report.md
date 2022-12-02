# Report

## ADAM: A Method for Stochastic Optimization

### Team: 43

- Priyanshul Govil


GitHub Link: 



# Submission

## Directory Structure

```

```

# Introduction

Optimizers play a very important role in the world of AI. They are the backbone of the learning processes which is as important as the loss function itself. In the language of commons, they tell the model what to learn from a given experience (experience being the loss). Gradient descent was the first-ever optimizer, introduced or rather suggested by Cauchy in 1847. Since the advent of backpropagation, AI picked up speed, and scientists have made attempts on a regular basis to
solve and find the best optimizers for fast and robust training. This lead to the development of various optimizers such as:

1. SGD
2. Momentum
3. Nesterov Accelerated GD
4. ADAGRAD
5. ADADELTA
6. RMSprop
7. ADAM

Each of them have a few variations too but are broadly classified above. All of these functions try to reach the minima of the loss function curve and how they do it is where they differ for example SGD has a constant learning rate and it tries to reach minima and it has a few issues too. Adam was proposed by considering the best of Adagrad and RMSprop and it solves issues raised because of them and is considered in general to be the best optimizer with minimal training loss
and calculations.

$m_t \leftarrow \beta_1 m_{t-1} + (1- \beta_1) g_t$
- Update biased first-moment eastimate (Equation 1)

$v_t \leftarrow \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- Update biased second raw moment estimate (Equation 2)

Equation 1 is from Adadelta which is the new learning parameter so the learning parameter is dependent on previous parameter and also current derivative.

Equation 2 is from RMSprop as we use a square of derivatives instead of derivatives for values to see the change in step.

The update step of Adam after scaling $m_t$ and $v_t$ appropriately using these equations

$m_t^1 \leftarrow m_t / (1 - \beta_1^t)$
- Compute bias-corrected first moment estimate

$v_t^1 \leftarrow v_t / (1 - \beta_2^t)$
- Compute bias-corrected second raw moment estimate

Update step:

$$
\theta_t \leftarrow \theta_{t-1} - \alpha \frac{m_t^1}{\sqrt{v_t^1} + \epsilon}
$$

In this study, we try to prove the claims made by the 2015 paper which introduced Adam. In addition to the general results, we also went a bit further and compared various optimizers and their performances on 3D polynomial terrains.

