import sys


class progressBar:
    def __init__(self, barWidth=50):
        self.barWidth = barWidth
        self.period = None

    def start(self, count):
        self.item = 0
        self.period = max(1, int(count / self.barWidth))
        sys.stdout.write("[" + (" " * self.barWidth) + "]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth + 1))

    def tick(self):
        if self.item > 0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1

    def stop(self):
        sys.stdout.write("]\n")
