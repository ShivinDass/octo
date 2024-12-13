import jax

# need a function like this: (start_batch, end_batch, sharding) -> Iterator[REPLAYBatch]
# * start batch: batch that you want to start iterating over
# * end batch: batch after the batch that you want to end iterating over (same as exclusive indexing)
# * sharding: dont worry about it just ignore and we'll figure it out later
# * (start_batch, end_batch) should return all the batches in order from start to end, exclusive for the end
# REPLAYBatch:
# * batch size
# * get_minibatches: part -> Iterator[REPLAYMinibatches]
# REPLAYMinibatches:
# * batch size
# * itself is an iterable that yields tuple[indices, (x, y)] for each minibatch

def naive_batch_maker(s, e, sharding, get_batch, minibs, num_batches):
    assert e <= num_batches
    num_devices = len(sharding.mesh.device_ids) if hasattr(sharding, 'mesh') else 1
    minibs = minibs * num_devices
    for i in range(s, e):
        batch = NaiveREPLAYBatch(get_batch(i), minibs, sharding)
        yield batch

class REPLAYBatch:
    def __init__(self, bs):
        # batch size of this batch
        self.bs = bs

    def get_minibatches(self, part):
        # part: 'train', 'val', or 'meta'
        raise NotImplementedError

class REPLAYMinibatches:
    def __init__(self, bs):
        self.bs = bs

    def __iter__(self):
        # iterates over minibatches, each minibatch is a (ixs, (x, y)) tuple
        # where ixs, x and y are jax arrays or numpy arrays
        # ixs: indices of the data points in the minibatch
        # x: input data
        # y: output data
        raise NotImplementedError

class NaiveREPLAYBatch(REPLAYBatch):
    def __init__(self, batch, minibs, sharding):
        bs = len(batch[0])
        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.sharding = sharding

    def get_minibatches(self, part):
        batch = jax.device_put(self.batch, self.sharding)
        minibs = int({
            'train': 1,
            'val': 1,
            'meta': 0.5
        }[part] * self.minibs)
        return NaiveREPLAYMinibatches(self.bs, batch, minibs=minibs)

def make_iterator(batch, minibs):
    s = 0
    this_bs = len(batch[0])
    while True:
        e = min(s + minibs, this_bs)
        ixs, (x, y) = batch
        sel = slice(s, e)
        s = e
        ixs, x, y = ixs[sel], x[sel], y[sel]
        yield ixs, (x, y)

        if e == this_bs:
            break

class NaiveREPLAYMinibatches(REPLAYMinibatches):
    def __init__(self, bs, batch, minibs):
        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs

    def __iter__(self):
        return make_iterator(self.batch, self.minibs)
