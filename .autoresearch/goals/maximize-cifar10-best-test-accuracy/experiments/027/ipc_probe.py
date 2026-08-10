import multiprocessing

from torch.utils.data import DataLoader


class SharedCollator:
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, batch):
        return batch, bool(self.flag.value)


def main():
    context = multiprocessing.get_context("forkserver")
    flag = context.Value("b", True, lock=True)
    loader = DataLoader(
        list(range(16)),
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context=context,
        collate_fn=SharedCollator(flag),
    )
    iterator = iter(loader)
    _batch, before = next(iterator)
    with flag.get_lock():
        flag.value = False
    observations = [before]
    for _ in range(7):
        try:
            _batch, seen = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            _batch, seen = next(iterator)
        observations.append(seen)
    iterator = None
    workers = list(loader._iterator._workers)
    loader._iterator._shutdown_workers()
    loader._iterator = None
    if observations[0] is not True or observations[-1] is not False:
        raise RuntimeError(f"flag propagation failed: {observations}")
    if any(worker.is_alive() for worker in workers):
        raise RuntimeError("worker shutdown failed")
    print(observations)


if __name__ == "__main__":
    main()
