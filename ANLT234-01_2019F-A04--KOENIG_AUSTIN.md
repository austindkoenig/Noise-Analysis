# <center>A Brief Noise Analysis</center>

<center>Austin Koenig</center>

## Introduction

Noise is the sole reason why every desired result about nature (i.e. the environment) is not deterministically calculatable. Unfortunately, we've yet to find a way to avoid it altogether; but, there exist many methods of removing noise from the observed data in order to obtain the signal of importance.

This is not a study on machine learning models. This is a study on noise and how it affects the performance of machine learning models with it's prominence in training data. First, we embark on a brief discussion on noise and it's origin; then, we consider some experimental results comparing the tolerance of noise between three different machine learning models.

### Noise Genesis

Before we define noise, let's set up a structure to the situation. First, let's assume our realm of conversation consists only of an agent (e.g. an engineer, robot, self-driving car, etc.) and the environment. Perhaps this environment contains other agents, or perhaps it contains some dynamic process which allows motion through time and space. In either case, we are only concerned with one agent within an environment.

Furthermore, let's precisely define an agent as an actor with a finite set of sensory inputs (e.g. eyes, a microphone, an infrared sensor, etc.) and a set of potential control outputs (e.g. raise arm, move forward ten meters, deploy parachute, etc.). 

In order for this agent to interact with the environment, she requres a way to measure things as well as some approximate knowledge of how the environment works. Formally, let's define these requirements as follows:

- Condition $A$ : An internal model<sup>1</sup> of the environment
- Condition $B$ : A method of observation 
- Condition $C$ : A method to combine environmental observations with the internal environmental model

It is also important to note that condition $C$ depends on conditions $A$ and $B$, which implies that should any of these conditions fail, we lose the assumption of agency. Therefore, an agent is bound to these conditions just as these conditions describe the very agency we've just defined.

Since we care to analyze the effect of noise in certain outcomes, this blog post is primarily concerned with conditions $B$ and $C$. As humans, our methods of observation include the systems involving our five primary senses. For self-driving cars, these methods of observation may include systems which rely on cameras, LIDAR sensors, and accelerometers. In any case, agents use these methods of observation to *measure* certain aspects of the environment. Intuitively, the aspects which an agent measures are chosen based on learned ideas from previous observations. 

The act of an agent measuring an environmental aspect is the point of noise genesis. Since the agent's internal model of the environment is an approximation of the environment itself, there exists the underlying assumption that agents cannot match their internal model exactly to the environment. The shorthand (and arguably more precise) version of this is to say that the agent's knowledge of the environment will always be *incomplete* due to the nondeterministic nature of known effective learning methods. 

This incompleteness is where noise comes from. Since each observation is only limited to the agent's set of sensory inputs, each observation yields only an incomplete version of the environment, which we will call the *local* environment (that is, the environmental aspects that are immediately observable by the agent). This incompleteness can propagate as the number of observations increases, especially if the agent has yet to consider any number of the components of the environment.

The remainder of this post will focus mainly on noise in machine learning methods since there is little that I can surely say about how humans learn.

<sup>1</sup> This usage of the term *model* is different than the common usage in machine learning. This usage refers to a general description outlining the components of the environment and how they all work.

### Motivation

Now that we know what noise is, it should be apparent why we want to avoid it.

## Data

The data used was generated from a simple sine wave. Noise was added to create training data and testing was performed directly on the function value.

### Noise

A brief discussion on noise in data.

### Training vs. Testing

Discuss the fact that we used lots of noise in the training sets but zero noise in the testing set. Here, we will show the plots of the testing data as well as from a few of the training sets as benchmarks.

## Models

We are comparing the following models to see which will be the least sensitive to noise:
1. Polynomial Regressor
2. Gradient Boost Regressor
3. Deep Neural Network

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