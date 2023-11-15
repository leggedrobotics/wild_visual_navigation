from torch.utils.data import Dataset
import torch

class VD_dataset(Dataset):
    def __init__(self, list_of_batches, combine_batches=False):
        # list_of_batches is a list of tuples: [(x1, y1), (x2, y2), ...]
        self.combine_batches = combine_batches
        # Dimension check
        for x, y in list_of_batches:
            if x.shape[0] != y.shape[0]:
                raise ValueError("Mismatch in batch size: x and y should have the same first dimension.")

        if combine_batches:
            # Combine all batches into one large batch
            xs, ys = zip(*list_of_batches)
            self.xs = torch.cat(xs, dim=0)
            self.ys = torch.cat(ys, dim=0)
        else:
            # Keep batches separate
            self.batches = list_of_batches

    def __len__(self):
        if self.combine_batches:
            return len(self.xs)
        else:
            return sum(len(batch[0]) for batch in self.batches)

    def __getitem__(self, index):
        if self.combine_batches:
            return self.xs[index], self.ys[index]
        else:
            # Find the right batch and index within that batch
            for x_batch, y_batch in self.batches:
                if index < len(x_batch):
                    return x_batch[index], y_batch[index]
                index -= len(x_batch)
            raise IndexError("Index out of range")

if __name__=="__main__":

    # Example usage
    # list_of_batches = [(torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)), ...]
    # dataset = VD_dataset(list_of_batches, combine_batches=True or False)
    batch_size1=10
    batch_size2=25
    x_dim=5
    y_dim=2
    list_of_batches = [(torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)), (torch.randn(batch_size2, x_dim), torch.randn(batch_size2, y_dim))]
    dataset = VD_dataset(list_of_batches, combine_batches=True or False)
    pass