![Banner](/Banner.png)

# About

`NURBS_PyTorch_2` is an implementation of NURBS geometries in [PyTorch](https://pytorch.org/). The flexible implementation allows for 10+ inputs (i.e. the dimensionality of the NURBS object) and an arbitrary number of outputs (i.e. the dimensionality of the space the NURBS object and its control points live in). PyTorch offers automatic differentiation so that these geometries are easily implemented in gradient descent and machine learning pipelines.

Interactive plotting is implemented with [JupyterDash](https://medium.com/plotly/introducing-jupyterdash-811f1f57c02e), and the other plots are made with Matplotlib.

See the notebooks for several examples.

# Solving differential equations

This NURBS implementation can be used to solve differential equations in a way inspired by [Physics-informed neural networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks) (PINNs): a neural network is trained to find the solution to a differential equation with suitable boundary conditions. Some examples of such solvers can be found in the notebooks.

The main differences between the approach presented here and PINNs are:
- The neural network itself is not the approximation to the DE solution, but its outputs define a NURBS object (e.g. the control points, weights, knots). 
- Therefore the solution approximation is not evaluated by evaluating the neural network but by evaluating the NURBS object.
- Derivatives of the NURBS object are hardcoded efficiently so that no backpropagation is needed to obtain these. Therefore backpropagation has to occur only once for an optimization iteration (also for higher derivative orders in the DE), from the loss to the neural network parameters.

# To do

- Implement derivatives of NURBS objects with weights
- Add surface area minimization example

# Disclaimer

I do not claim that this code is correct or reliable, please check it against an established B-spline/NURBS implementation for your application.
