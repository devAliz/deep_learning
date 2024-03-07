import random

seed = 400121030
random.seed(seed)
vector = [random.randint(0, 20) for _ in range(10000)]
counts = [0] * 21
for num in vector:
  counts[num] += 1
for i in range(21):
  print(f"{i}: {counts[i]}")