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

For reasons like this, it is important to combat noise to the fullest of our abilities. As stated above, noise is caused by a disparity between the perspectives of interacting agents. Later in this post, we will discuss an experiment comparing a few different machine learning models and their abilities to find the signal in the noise. First, we will see how the data was generated and then describe the models which were tested. Finally, we will go over the results of the experiment.

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

In English, this means that all input values are real numbers between $-2\pi$ and $2\pi$; the testing outputs are the exact sine value, and the training outputs are samples from the normal distribution with mean equal to the exact sine value and a varying standard deviation. The standard deviation varies across what we will call *degrees of noise*, which ranges from 1.0.

## Models

We are comparing the following models to see which will be the least sensitive to noise:
1. Polynomial Regressor
2. Gradient Boost Regressor
3. Deep Neural Network

### What is a Polynomial Regressor?

Briefly explain what a polynomial regressor is and why we chose this model.

### What is a Gradient Boost Regressor?

Briefly explain what a gradient boosting is and why we chose this model.

### What is a Deep Neural Network?

Briefly explain what a deep neural network is and why we chose it.

## Results & Discussion

We found that Model 1 fell victim to excess amounts of noise where Model 2 was able to withstand the noise rather well.

Here, we will show the prediction plots and loss plots in order to outline the performance of our model. While looking at a table of error values can be convincing, a picture is worth a thousand words--or numbers, in this case.

We will also highlight a rather contraversial observation of our results.

## Conclusion

Discuss the further implications of these results. If we have relatively clean data, we are more likely to be able to get away with using models which rely on fewer parameters. However, if there is a lot of noise in our data, then a safer solution would be a deep neural network of some kind.

## Sources

[1] [Organism Fact Sheet: Birds of Paradise](https://www.famsf.org/files/Fact%20Sheet_%20birds_of_paradise.pdf)