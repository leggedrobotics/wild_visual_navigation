from torch.utils.data import Dataset
import torch

class VD_dataset(Dataset):
    def __init__(self, list_of_batches, combine_batches=True):
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
            self.batches=[(self.xs,self.ys)]
        else:
            # Keep batches separate
            self.batches = list_of_batches

    def __len__(self):
        return sum(len(batch[0]) for batch in self.batches)

    def __getitem__(self, index):
        # Find the right batch and index within that batch
        for x_batch, y_batch in self.batches:
            if index < len(x_batch):
                return x_batch[index], y_batch[index]
            index -= len(x_batch)
        raise IndexError("Index out of range")
    
    def get_batch_num(self):
        return len(self.batches)

    def get_x(self,batch_idx=None):
        if self.combine_batches:
            return self.xs
        else:
            if batch_idx is None:
                raise ValueError("batch_idx must be specified when batches are not combined.")
            return self.batches[batch_idx][0]
    
 
    def get_y(self,batch_idx=None):
        if self.combine_batches:
            return self.ys
        else:
            if batch_idx is None:
                raise ValueError("batch_idx must be specified when batches are not combined.")
            return self.batches[batch_idx][1]
    
    def add_batches(self, new_list_of_batches):
        # Perform dimension check as before
        for x, y in new_list_of_batches:
            if x.shape[0] != y.shape[0]:
                raise ValueError("Mismatch in batch size: x and y should have the same first dimension.")

        if self.combine_batches:
            # If batches are combined, just concatenate the new data
            new_xs, new_ys = zip(*new_list_of_batches)
            self.xs = torch.cat([self.xs] + list(new_xs), dim=0)
            self.ys = torch.cat([self.ys] + list(new_ys), dim=0)
            self.batches=[(self.xs,self.ys)]
        else:
            # If batches are separate, extend the list
            self.batches.extend(new_list_of_batches)
    
    def save_dataset(dataset, file_path):
        torch.save(dataset, file_path)
    
    def load_dataset(file_path):
        return torch.load(file_path)



if __name__=="__main__":

    # Example usage
    # list_of_batches = [(torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)), ...]
    # dataset = VD_dataset(list_of_batches, combine_batches=True or False)
    batch_size1=10
    batch_size2=25
    x_dim=5
    y_dim=2
    list_of_batches = [(torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)), (torch.randn(batch_size2, x_dim), torch.randn(batch_size2, y_dim))]
    dataset = VD_dataset(list_of_batches, combine_batches=True)
    ss=dataset[1]
    batch_size1=20
    batch_size2=35
    x_dim=5
    y_dim=2
    list_of_batches = [(torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)), (torch.randn(batch_size2, x_dim), torch.randn(batch_size2, y_dim))]
    dataset.add_batches(list_of_batches)
    pass