# <center>A Brief Noise Analysis</center>

<center>Austin Koenig</center>

## Introduction

Noise is the reason why every desired result about nature (i.e. the environment) is not deterministically calculatable. Unfortunately, we've yet to find a way to avoid it altogether; but, there exist many methods of removing noise from the observed data in order to obtain the signal of importance

This is not a study on machine learning models, but a study on noise and how it affects the performance of machine learning models with it's prominence in training data. First, we embark on a brief discussion on noise and it's origin; then, we consider some experimental results comparing the tolerance of noise between three different machine learning models.

## Noise Genesis

Before we define noise, let's prepare a platform off of which to jump towards the idea of noise. 

First, let's assume our realm of conversation consists only of an agent (e.g. an engineer, robot, self-driving car, etc.) and the environment. Perhaps this environment contains other agents, or perhaps it contains some dynamic processes which allow motion through time and space. In either case, we are only concerned with one agent within an environment.

Furthermore, let's describe an agent as an actor with a set of sensory inputs (e.g. eyes, a microphone, an infrared sensor, etc.) and a set of potential control outputs (e.g. raise arm, move forward ten meters, deploy parachute, etc.). 

In order for this agent to interact with the environment, some other conditions must be met. Let's formally define an agent as follows.

An agent is an actor (e.g. human, robot, car, etc.) that possesses
1. A set of sensory inputs
2. A set of control outputs

Moreover, in order for agency to exist, the following conditions must hold:
- $A$ : The agent maintains an internal model<sup>1</sup> of the environment.
- $B$ : The agent has at least one method of observation.
- $C$ : The agent is able to combine environmental observations with the internal environmental model.

Since we care to analyze the effect of noise in certain outcomes, this blog post is primarily concerned with condition $B$ and somewhat concerned with condition $C$. As humans, our methods of observation include the systems involving our primary senses. For self-driving cars, these methods of observation may include systems which rely on cameras, LIDAR sensors, and accelerometers. In any case, agents use these methods of observation to *measure* certain aspects of the environment. Intuitively, the aspects which an agent measures are chosen based on learned ideas from previous observations. From this, we assume that the agent has no prior knowledge of the environment such that they have to learn it from pure observation and measurement.

The act of an agent measuring an environmental aspect is the point of noise genesis. Since the agent's internal model of the environment is an approximation of the environment itself that is only based on observation, there exists the underlying assumption that agents cannot map their internal model exactly to the environment. The shorthand (and more precise) version of this is to say that the agent's knowledge of the environment will always be *incomplete* due to the nondeterministic nature of known effective learning methods.

This incompleteness is the origin of noise. Since each observation is only limited to the agent's set of sensory inputs, each observation yields only an incomplete version of the environment, which we will call the *local* environment (that is, the environmental aspects that are immediately observable by the agent). The noise is defined as the difference between the local environment and the global environment. This incompleteness can propagate as the number of observations increases, especially if the agent has yet to consider any number of the components of the environment.

Why is noise a problem? One might argue that the agent only needs to be concerned with her local environment because it is the only part of the environment that proposes interactibility to the agent. However, this perspective is skewed: while this post has thus far only discussed the situation of one agent in an environment, reality consists of many agents across many disjoint local environments. Thus, agents must avoid conflict with each other by considering what is beyond each of their local environments.

For instance, both humans and birds of paradise<sup>2</sup> are agents according to our definition. On the island of New Guinea, these agents are in conflict. People are destroying the birds' natural habitats in order to harvest and grow resources, thereby threatening the existence of these beautiful creatures [1]. Humans are concerned with their local environment which requires their possession of crops and building materials whereas birds of paradise are concerned with the survival of their local environment. There is a great amount of noise stemming from these disjoint local environments, culminating in the potential destruction of a species.

For reasons like this, it is important to combat noise to the fullest of our abilities. As stated above, noise is caused by a disparity between the perspectives of interacting agents. Later in this post, we will discuss an experiment comparing a few different machine learning models and their abilities to find the signal in the noise. First, we will see how the data was generated and then briefly describe the models which were tested. Finally, we will discuss the results of the experiment.

<sup>1</sup> This usage of the term *model* is different than the common usage in machine learning. Here, it refers to a general description outlining the components of the environment and how they all work.

<sup>2</sup> I highly recommend the Netflix documentary called [Dancing With The Birds](https://www.netflix.com/title/80186796).

## Data

The data used was generated from a sine wave. A sine wave was chosen because it is a simple function that isn't a polynomial. This is an important feature of the experiment because one of our models will employ polynomials, so we don't want there to be a "competitive advantage".

Following is a brief description on how each of the sets of data were calculated.

The input set for both training and testing is 

- Training & Testing Inputs:
  $$X=\{x\in\mathbb{R}[-2\pi, 2\pi]\}$$ 
- Testing Outputs: 
  $$Y_{test}=\{\sin{x}\;|\; x\in X\}$$
- Training Outputs: 
  $$Y_{train}=\{\mathcal{N}(y, \sigma^2);|\; y\in Y_{test},\;\sigma^2\in\varSigma\}$$
  where $\varSigma$ is the set of all "noise levels".

In English, this means that all input values are real numbers between $-2\pi$ and $2\pi$; the testing outputs are the exact sine value, and the training outputs are samples from the normal distribution with mean equal to the exact sine value and a varying standard deviation. The standard deviation varies across what we will call *degrees of noise*, which ranges between 0.001 and 1.

The Python package `numpy` offers a few different ways we can range through the interval $[0.001, 1]$. The two we are interested in are:
- [`numpy.linspace`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html) - Linear spacing means that the points that are sampled are equidistant from each other.
- [`numpy.geomspace`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.geomspace.html) - Geometric spacing means that the points that are sampled are equidistant from each other on a log scale.

The reason this matters is that if we use geometric (or logarithmic) spacing, then the sampled points will be concentrated in a nonlinear fashion. To illustrate this more clearly, consider the following plot:

![Noise Plot](./results/figures/noise.pdf)

Notice that the points sampled linearly create a straight line whereas the points sampled geometrically are concentrated more in the lower values even though they vary across the same interval. If we were concerned more about situations with less noise, perhaps the geometrically spaced samples would suit us; however, we wish to be robust and simply use the linearly spaced points.

Overall, we have two sets: training and testing. Our training set has one set of inputs and twelve sets of outputs (we experiment with 12 degrees of noise). The testing set has one set of inputs and one set of outputs. The goal is to create and test a separate model (of each variety described below) for each degree of noise. Then, we will compare the errors and predictions of each type of model to see how they withstand the noise that we've just generated.

## Models

We are comparing the following models to see which will be the least sensitive to noise:
1. Polynomial Regressor - A polynomial regressor is an extension of linear regression in that it employs higher order terms to make predictions.
2. Gradient Boost Regressor - A gradient boost regressor is an extension of decision trees (or random forests) in that it uses gradient descent for parameter optimization.
3. Shallow Neural Network - Shallow neural networks can be considered as a few layers of nonlinear combinations
4. Deep Neural Network - Deep neural networks are similar to shallow neural networks; they simply contain more than one hidden layer.

This group of models was chosen because it includes one model that doesn't use gradient descent, two models which are relatively small by number of parameters, and one deep neural network that is presumably the most accurate model. This mix seems to encapsulate a fair amount of the varieties of learning models that are employed. How did these models actually perform?

## Results & Discussion

The most obvious thing to look at are the errors. Since the models we picked are relatively different in the number of parameters. This will cause an inherent difference in errors, but we wish only to observe the trends with respect to one another. Therefore, they have been scaled to a standard normal distribution so that we can observe them all side-by-side.

![Error Plot](./results/figures/errors.pdf)

While knowing the error is useful in giving a rating system for the models, it is still very helpful to see the prediction plots for each model. For instance, techniques like high-order Newton-Cotes interpolation operate very poorly around the edges of the interval. This is not reflected in the error, so we can use the prediction plots to see in which areas of the function to be learned (sine, in this case) our models performed the best/worst. Moreover, we can see how these behaviors change as we introduce more and more noise into the data.

![Prediction Plot of 0 Degrees of Noise](./results/figures/prediction--noise-0.pdf)

![Prediction Plot of 4 Degrees of Noise](./results/figures/prediction--noise-4.pdf)

![Prediction Plot of 8 Degrees of Noise](./results/figures/prediction--noise-8.pdf)

![Prediction Plot of 12 Degrees of Noise](./results/figures/prediction--noise-12.pdf)

## Conclusion

There are many methods to preprocess data to sift through noise, none of which we used here. This post was focused mainly on how particular types of machine learning models withstand the burden of noise. We tested four different models, which were meant to be archetypes for different machine learning methods, against sixteen different degrees of noise. There are techniques to filter data in nearly every stage of the data science process. We have only observed the capabilities of models in handling the noise themselves. In reality, much more data cleaning and preprocessing would have occured, but much of that was omitted due to the nature of the experiment.

Going forward, we can look at other filtering methods that are not embedded directly into the model itself. Also, a wider range of models and model sizes should be tested. There is even potential for a search problem in finding the best combination of noise reduction techniques using a machine learning model. However, this blog is a good start and poses a platform on which to build even deeper ideas about dealing with noise in data. We have simply withstood it, but in the future we wish to reduce it before it even reaches the models.

## Sources

[1] [Organism Fact Sheet: Birds of Paradise](https://www.famsf.org/files/Fact%20Sheet_%20birds_of_paradise.pdf)