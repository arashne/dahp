from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, log_dir, logging_freq=1):
        self.log_dir = log_dir
        self.logging_freq = logging_freq

        self.summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

        print(f'logging outputs to {self.log_dir}')

    def log(self, key: str, val, step: int):
        if step % self.logging_freq == 0:
            self.summ_writer.add_scalar(key, val, step)
            return True
        else:
            return False

    def flush(self):
        self.summ_writer.flush()
