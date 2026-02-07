# Grandma's Computer Class: Understanding JAX Code

Welcome to JAX! This guide explains how the code works in the simplest way possible, using everyday examples you can relate to.

## What is JAX?

Think of JAX as a super-powered calculator for computers. Just like your calculator at home can do math quickly, JAX helps computers do really big math problems *very* fast. It's especially good at:

1. **Doing the same math many times** - Like if you had to calculate your grocery bill 1,000 times
2. **Making things faster** - Like having a faster calculator
3. **Learning from mistakes** - Like practicing and getting better at something

## Understanding the MNIST Handwritten Digit Example

Let's look at a program that teaches a computer to read handwritten numbers (0-9), just like you learned to recognize numbers as a child!

### The Big Picture (What We're Doing)

Imagine teaching a child to recognize numbers:
1. You show them many examples of handwritten numbers
2. They guess what each number is
3. You tell them if they're right or wrong
4. They learn from their mistakes and get better

Our computer program does the exact same thing!

### The Code, Explained Simply

#### Step 1: Setting Up the "Brain" (Neural Network)

```python
def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]
```

**What's happening here?**
- We're creating the computer's "brain" with random starting knowledge
- Think of it like a baby - at first, it doesn't know anything
- `layer_sizes = [784, 1024, 1024, 10]` means:
  - **784**: The computer looks at 784 tiny dots (pixels) in each number image (28×28 = 784)
  - **1024, 1024**: Two "thinking" layers where the computer processes what it sees
  - **10**: Final answer layer with 10 choices (digits 0-9)

#### Step 2: Making Guesses (Predictions)

```python
def predict(params, inputs):
  activations = inputs
  for w, b in params[:-1]:
    outputs = jnp.dot(activations, w) + b
    activations = jnp.tanh(outputs)
  
  final_w, final_b = params[-1]
  logits = jnp.dot(activations, final_w) + final_b
  return logits - logsumexp(logits, axis=1, keepdims=True)
```

**What's happening here?**
- The computer looks at an image and guesses what number it is
- It's like showing a picture to a child and asking "What number is this?"
- The computer processes the image through several steps (layers) of thinking
- At the end, it gives you 10 numbers - one for each possible digit (0-9)
- The highest number is its best guess

**Simple Analogy:**
Think of it like a detective looking at clues:
1. First, they see the clues (the image)
2. Then they think about what they mean (processing through layers)
3. Finally, they make their best guess about whodunit (which digit it is)

#### Step 3: Checking If the Guess Was Right (Loss Function)

```python
def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))
```

**What's happening here?**
- We compare the computer's guess to the right answer
- It's like grading a test - how many did it get wrong?
- The "loss" is like counting mistakes
- **Lower loss = Fewer mistakes = Better!**

**Simple Analogy:**
If you took a spelling test:
- You write down your answers (predictions)
- Teacher shows you the correct answers (targets)
- Teacher counts how many you got wrong (loss)
- Your goal is to get fewer wrong next time!

#### Step 4: Measuring Success (Accuracy)

```python
def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)
```

**What's happening here?**
- We count how many the computer got exactly right
- Like scoring a test: "You got 85 out of 100 correct!"
- This gives us a percentage, like 85% accuracy

#### Step 5: Learning from Mistakes (Training)

```python
@jit
def update(params, batch):
  grads = grad(loss)(params, batch)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]
```

**What's happening here?**
- This is where the computer learns and gets better!
- `grad(loss)` figures out *how* to improve
- It's like a teacher saying "To do better, you should study this part more"

**Simple Analogy:**
Think of learning to throw darts:
1. You throw and miss the bullseye
2. Someone tells you "You threw too far to the left"
3. Next time, you aim a bit more to the right
4. You keep adjusting until you hit the bullseye

The `@jit` decorator is like having a coach who makes you practice the same motion faster and faster until it becomes automatic.

#### Step 6: Practice Makes Perfect (The Main Training Loop)

```python
for epoch in range(num_epochs):
  start_time = time.time()
  for _ in range(num_batches):
    params = update(params, next(batches))
  epoch_time = time.time() - start_time
  
  train_acc = accuracy(params, (train_images, train_labels))
  test_acc = accuracy(params, (test_images, test_labels))
  print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
  print(f"Training set accuracy {train_acc}")
  print(f"Test set accuracy {test_acc}")
```

**What's happening here?**
- The computer practices over and over again
- **epoch**: One complete practice session where it sees all the training examples
- **num_epochs = 10**: We practice 10 times through all the examples
- After each practice session, we test how well it's doing

**Simple Analogy:**
Like learning your multiplication tables:
- **Day 1**: You practice all the tables once and get 60% right
- **Day 2**: You practice again and get 70% right  
- **Day 3**: You practice again and get 80% right
- After 10 days of practice, you get 95% right!

### The Settings Explained

```python
layer_sizes = [784, 1024, 1024, 10]  # Size of the "brain"
param_scale = 0.1                     # How random the starting guesses are
step_size = 0.001                     # How big each learning step is
num_epochs = 10                       # How many times to practice
batch_size = 128                      # How many examples to look at together
```

**Simple Analogies:**
- **layer_sizes**: Like deciding how many pages are in your study notebook
- **param_scale**: Like starting with small random guesses instead of wild guesses
- **step_size**: Like taking small careful steps vs big jumps when learning
  - Too big = You might overshoot and never get it right
  - Too small = It takes forever to learn
- **num_epochs**: Like deciding to read through your textbook 10 times
- **batch_size**: Like studying 128 flashcards before checking your answers

## Key Concepts in Simple Terms

### What is "jit"?
The `@jit` decorator is like teaching the computer a shortcut. 

**Without @jit**: "Take out calculator, press 2, press +, press 2, press ="
**With @jit**: "I know you're going to do this a lot, so memorize this pattern and do it faster next time!"

### What are "gradients"?
Gradients tell you which direction to go to get better.

**Analogy**: You're blindfolded trying to find the lowest point in a valley:
- Gradients tell you "go downhill this way"
- You take a small step in that direction
- Check again, and repeat
- Eventually, you find the bottom!

### What is "batch_size"?
Instead of learning from one example at a time, we show the computer many examples together.

**Analogy**: 
- Learning from 1 photo = Looking at just one flashcard
- Learning from 128 photos (batch) = Looking at a stack of flashcards and learning the general pattern

This is faster and gives better overall understanding!

## Summary: The Whole Process

1. **Start**: Computer knows nothing (random brain)
2. **Show**: Show it examples of handwritten numbers
3. **Guess**: Computer guesses what each number is
4. **Check**: We tell it if it's right or wrong
5. **Learn**: Computer adjusts its "brain" to do better
6. **Repeat**: Do this thousands of times
7. **Result**: Computer can now read handwritten numbers really well!

After 10 practice sessions (epochs), the computer typically learns to recognize handwritten digits with 95%+ accuracy - better than many humans!

## Common Questions

**Q: Why does it need so much practice?**
A: Just like you needed to practice reading as a child! The computer starts with zero knowledge and builds up understanding through repetition.

**Q: What does the computer actually "see"?**
A: It sees 784 numbers (one for each pixel). Dark pixels = big numbers, light pixels = small numbers. So a "7" might look like a specific pattern of 784 numbers.

**Q: How does it actually learn?**
A: It uses math to figure out "if I change this number in my brain by a tiny bit, do I do better or worse?" Then it changes all its brain numbers to do better, little by little.

**Q: Why is it called "from scratch"?**
A: Because we're building everything ourselves - the brain structure, the learning process, everything! It's like baking a cake from scratch vs using a cake mix.

## Final Thoughts

Machine learning is really just:
1. Making guesses
2. Checking if you're right
3. Learning from mistakes
4. Repeating until you get good

It's exactly how you learned everything in life - from walking, to talking, to reading! The computer is doing the same thing, just with math instead of neurons.

Welcome to the world of machine learning - you now understand how it works! 🎉
