import math

class WeightScheduler:
    def __init__(self, weight_max: float, total_epochs: int, zero_epochs: int = 5, warmup_epochs: int = 10):
        """
        S-shaped growth scheduler for weight with an initial low period.

        :param weight_max: The maximum weight at the end of warm-up.
        :param total_epochs: The total number of epochs for training.
        :param zero_epochs: The number of epochs the weight remains zero.
        :param warmup_epochs: The number of epochs over which the weight increases.
        """
        self.weight_max = weight_max
        self.total_epochs = total_epochs
        self.zero_epochs = zero_epochs
        self.warmup_epochs = warmup_epochs
        self.last_epoch = 0

    def get_weight(self):
        """
        Calculate the weight using a modified sigmoid growth function.
        :return: The current weight value.
        """
        # Before the warm-up phase, keep the weight low
        if self.last_epoch < self.zero_epochs:
            weight = 0.0
        else:
            # Calculate the progress in the warm-up phase
            progress = (self.last_epoch - self.zero_epochs) / self.warmup_epochs
            # Sigmoid function to generate S-shaped curve
            weight = self.weight_max / (1 + math.exp(-10 * (progress - 0.5)))

            # Ensure weight does not exceed weight_max
            weight = min(weight, self.weight_max)

        return weight

    def step(self):
        """
        Increment the epoch count for KL weight scheduling.
        """
        self.last_epoch += 1


if __name__ == "__main__":
    weight_scheduler = WeightScheduler(weight_max=1.0, total_epochs=100, zero_epochs=5, warmup_epochs=10)
    for i in range(100):
        print(weight_scheduler.get_weight())
        weight_scheduler.step()