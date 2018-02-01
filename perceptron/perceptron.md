# perceptron
---

#### what is it?

Basically its a simulation of a single neuron, or at least thats the most common anology there is for them. On a lower level, they take a function $C_1x + C_2y + C_3z ... + C_nN = C_0$ and a lot of points with classes (1 or 0) as input. Then they fine tune the $C_n$s to the values that best draw a line between the points classes.

#### what are points classes?

Basically, a class is just a 0 or a 1, a true or a false, you get the idea. Imagine you are eating pizza from a hundred different pizza places, you dont really need a fancy rating system to tell if you want to go there again or not, just classify the restauraunt as having "good pizza" or "bad pizza". Those are your two classes.

#### what was that equation?

So there is really not that much scary math in perceptrons, its no more complex than what you learned in elementary school. Lots of papers online are full of complex equations like $C_1x + C_2y + C_3z ... + C_nN = C_0$ but this is actually quite simple. You have seen something like this before... Remember $ax + by = C$? The equation for a line? Thats the *exact same thing* as what we are using, just using 2 dimensions instead of the more generalized form I wrote above. When N is 2, it becomes $C_1x + C_2y = C_0$ look less scary now?

So basically our perceptron is just going to find a line between the two classes we give it ("good pizza" and "bad pizza"). Then when we plug in new data, the perceptron will output 1 if the point is above the line ("good pizza") and 0 if it is below the line ("bad pizza").

#### how does it find the best line?

This is the most complex part of how perceptrons work, so if you can get through this, you basically are good for the rest of it. 

So we are giving it a lot of points ${(x_1, y_1), (x_2, y_2), (x_3, y_3)...(x_n,y_n)}$ as input, and also telling it the class of each point ${p_1, p_2, p_3...p_n}$. The perceptron will look at every case one by one and apply its learning rule (more on that next). For now, just understand that it will look at $(x_1, y_1)$ and $p_1$ and modify its rule accordingly using its **learning algorithm**.

#### what is its learning algorithm?

Its fairly simple, bascially, for every group of input and output...

1. guess. Using the current line, try to predict if the given point is "good pizza" or "bad pizza"
2. find the error. Take the correct answer and subtract the predicted value. We will call this $E$
3. for every weight (Those are what we called $C_n$ in the above equations) add the error $E$ multiplied by the input value coresponding to that variable, multiplied by a **learning rate** (more on that later). 

	Basically, $C_n' = C_n + (E * L * I_n)$
	
	Here $C_N'$ is the newly updated $C_n$, $E$ is the error, $I_n$ is the input for that variable, and $L$ is the **learning rate**. 
	
	This part might be a little confusing, so lets break it down. We want to compute the change in the weight, so what things are important here? Well the error definitly is, if we were correct, we dont need to change at all, but if we were wrong 