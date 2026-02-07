#!/usr/bin/env python3
# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GRANDMA'S COMPUTER CLASS: Learning to Add Numbers with JAX

This is the SIMPLEST possible example to understand how JAX works.
We're going to teach the computer to learn the function: f(x) = 2*x + 3

Think of it like this:
- You give me a number (input)
- I secretly multiply it by 2 and add 3 (the rule)
- I tell you the result
- Can you guess my secret rule?

That's what the computer will learn to do!
"""

import jax
import jax.numpy as jnp
from jax import grad, jit

print("=" * 60)
print("GRANDMA'S COMPUTER CLASS: Teaching a Computer to Learn")
print("=" * 60)
print()

# ============================================================================
# STEP 1: Create some example data
# ============================================================================
print("STEP 1: Making Some Examples")
print("-" * 40)

# Let's say we have a secret rule: y = 2*x + 3
# We'll make some examples where we know the answer
examples_x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Input numbers
examples_y = jnp.array([5.0, 7.0, 9.0, 11.0, 13.0])  # Correct answers

print("Here are our teaching examples:")
print("Input  -> Output")
for x, y in zip(examples_x, examples_y):
    print(f"{x:5.1f} -> {y:5.1f}")

print()
print("Can you see the pattern?")
print("(Hint: Each output is the input times 2, plus 3)")
print()

# ============================================================================
# STEP 2: Create the computer's "brain" (the parameters it will learn)
# ============================================================================
print("STEP 2: Setting Up the Computer's Brain")
print("-" * 40)

# The computer needs to learn two numbers:
# - The multiplier (we want it to learn 2.0)
# - The amount to add (we want it to learn 3.0)
#
# But we start with random guesses - the computer doesn't know yet!

key = jax.random.PRNGKey(0)  # Random number generator
weights = jax.random.normal(key, (2,))  # Two random starting numbers

# Let's call them 'slope' and 'intercept' to be clear
slope = weights[0]
intercept = weights[1]

print(f"Computer's starting guess for the multiplier: {slope:.4f}")
print(f"Computer's starting guess for the add amount: {intercept:.4f}")
print()
print("(These are random - the computer doesn't know the answer yet!)")
print()

# ============================================================================
# STEP 3: Make the computer's prediction function
# ============================================================================
print("STEP 3: How the Computer Makes Guesses")
print("-" * 40)

def predict(params, x):
    """
    This is the computer's guess at the answer.
    
    Simple analogy: If someone asks "what's 2 times 7?"
    - You multiply: 2 * 7 = 14
    
    Here, the computer multiplies x by its guess for 'slope',
    then adds its guess for 'intercept'.
    
    params: The computer's current best guesses [slope, intercept]
    x: The input number
    """
    slope, intercept = params
    prediction = slope * x + intercept
    return prediction

# Let's see what it predicts with its random guesses
test_input = 2.0
test_prediction = predict(weights, test_input)
correct_answer = 7.0

print(f"If we give the computer the number {test_input}")
print(f"It predicts: {test_prediction:.4f}")
print(f"The correct answer is: {correct_answer}")
print(f"That's an error of: {abs(test_prediction - correct_answer):.4f}")
print()
print("Not very good! But it will learn...")
print()

# ============================================================================
# STEP 4: Measure how wrong the computer is
# ============================================================================
print("STEP 4: Measuring Mistakes")
print("-" * 40)

def loss_function(params, x_data, y_data):
    """
    This measures how wrong the computer is.
    
    Simple analogy: If you took a test with 5 questions:
    - You got: [5, 7, 8, 11, 12]
    - Correct: [5, 7, 9, 11, 13]
    - Errors: [0, 0, 1, 0, 1] -> Average error = 0.4
    
    We square the errors to make big mistakes count more.
    This is called "Mean Squared Error" (MSE).
    
    params: Computer's guesses [slope, intercept]
    x_data: Input numbers
    y_data: Correct answers
    """
    predictions = predict(params, x_data)
    errors = predictions - y_data
    squared_errors = errors ** 2
    average_squared_error = jnp.mean(squared_errors)
    return average_squared_error

# Check how bad the initial guess is
initial_loss = loss_function(weights, examples_x, examples_y)
print(f"The computer's initial mistake score: {initial_loss:.4f}")
print("(Lower is better - 0 would be perfect!)")
print()

# ============================================================================
# STEP 5: Figure out how to improve (this is the magic!)
# ============================================================================
print("STEP 5: Learning How to Improve")
print("-" * 40)

# This is where JAX's magic happens!
# The 'grad' function figures out: "How should I change my guesses
# to make fewer mistakes?"
#
# It's like a teacher telling you:
# "To do better on the test, study THIS part more"

gradient_function = grad(loss_function)

# Calculate the gradients (directions to improve)
gradients = gradient_function(weights, examples_x, examples_y)

print("JAX calculated the 'gradients' (how to improve):")
print(f"Change to multiplier suggestion: {gradients[0]:.4f}")
print(f"Change to add amount suggestion: {gradients[1]:.4f}")
print()
print("Negative gradients mean 'increase this number'")
print("Positive gradients mean 'decrease this number'")
print()

# ============================================================================
# STEP 6: Actually improve (learning step)
# ============================================================================
print("STEP 6: Taking a Learning Step")
print("-" * 40)

learning_rate = 0.01  # How big of a step to take

# Take a small step in the direction that reduces mistakes
new_weights = weights - learning_rate * gradients

new_slope = new_weights[0]
new_intercept = new_weights[1]

print(f"Old multiplier guess: {slope:.4f} -> New: {new_slope:.4f}")
print(f"Old add amount guess: {intercept:.4f} -> New: {new_intercept:.4f}")
print()

# Check if we improved
new_loss = loss_function(new_weights, examples_x, examples_y)
print(f"Old mistake score: {initial_loss:.4f}")
print(f"New mistake score: {new_loss:.4f}")
print(f"Improvement: {initial_loss - new_loss:.4f} (we got better!)")
print()

# ============================================================================
# STEP 7: Repeat many times (this is training!)
# ============================================================================
print("STEP 7: Practice Makes Perfect!")
print("-" * 40)
print("Now we'll repeat this process 1000 times...")
print()

# Use @jit to make this fast (JAX compiles it)
@jit
def update_step(params, x_data, y_data, learning_rate):
    """One complete learning step: calculate gradients and update."""
    grads = grad(loss_function)(params, x_data, y_data)
    new_params = params - learning_rate * grads
    return new_params

# Start fresh
params = jax.random.normal(key, (2,))
learning_rate = 0.01

# Train for 1000 steps
print("Step  |  Multiplier  |  Add Amount  |  Mistake Score")
print("-" * 60)

for step in range(1001):
    params = update_step(params, examples_x, examples_y, learning_rate)
    
    # Print progress every 200 steps
    if step % 200 == 0:
        current_loss = loss_function(params, examples_x, examples_y)
        print(f"{step:4d}  |   {params[0]:8.4f}   |   {params[1]:8.4f}   |   {current_loss:.6f}")

print()
print("TRAINING COMPLETE!")
print("=" * 60)

# ============================================================================
# STEP 8: See what the computer learned
# ============================================================================
print()
print("STEP 8: What Did the Computer Learn?")
print("-" * 40)

final_slope = params[0]
final_intercept = params[1]

print(f"The computer learned:")
print(f"  Multiplier: {final_slope:.4f} (should be close to 2.0)")
print(f"  Add amount: {final_intercept:.4f} (should be close to 3.0)")
print()
print("Let's test it on the examples:")
print()
print("Input  |  Computer's Answer  |  Correct Answer  |  Error")
print("-" * 60)

for x, y_true in zip(examples_x, examples_y):
    y_pred = predict(params, x)
    error = abs(y_pred - y_true)
    print(f"{x:5.1f}  |      {y_pred:8.4f}       |      {y_true:5.1f}       |  {error:.6f}")

print()
print("=" * 60)
print("CONGRATULATIONS!")
print("You just taught a computer to learn a mathematical pattern!")
print("=" * 60)
print()
print("KEY TAKEAWAYS:")
print("1. The computer started knowing NOTHING (random guesses)")
print("2. We showed it examples of the pattern")
print("3. It made guesses and we measured the mistakes")
print("4. JAX calculated how to improve (gradients)")
print("5. The computer adjusted its guesses little by little")
print("6. After 1000 practice rounds, it learned the pattern!")
print()
print("This is EXACTLY how all machine learning works:")
print("  - Make a guess")
print("  - Measure the mistake")
print("  - Learn from the mistake")
print("  - Repeat until you're good at it")
print()
print("Now you understand JAX and machine learning!")
print("=" * 60)
