import torch

class EmotionalOptimizer:
    def __init__(self, optimizer, personality: str = "wholesome"):
        self.optimizer = optimizer
        self.personality = personality

    def step(self, *args, **kwargs):
        # TODO: add emotional logging later
        return self.optimizer.step(*args, **kwargs)
