"""Iteration scheduler for periodic closed-loop diagnostics."""


class ClosedLoopScheduler:
    def __init__(self, start_iter=1000, interval=500, max_triggers=-1):
        self.start_iter = int(start_iter)
        self.interval = max(1, int(interval))
        self.max_triggers = int(max_triggers)
        self.trigger_count = 0

    def should_trigger(self, iteration):
        if iteration < self.start_iter:
            return False
        if self.max_triggers >= 0 and self.trigger_count >= self.max_triggers:
            return False
        return (int(iteration) - self.start_iter) % self.interval == 0

    def mark_triggered(self):
        self.trigger_count += 1
