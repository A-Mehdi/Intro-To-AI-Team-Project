import torch.utils.data as data
from PIL import Image
import os
import os.path

def has_file_allowed_extension(filename, extensions):
    file_lower = filename.lower()
    return any(file_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    classes_to_id_ = {classes[i]: i for i in range(len(classes))}
    return classes, classes_to_id_


def make_dataset(dir, extensions):
    imgs = []
    for root, _, filenames in sorted(os.walk(dir)):
        for fname in sorted(filenames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                imgs.append(item)

    return imgs

class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, targets = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return sample, targets

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        format_str = 'Dataset ' + self.__class__.__name__ + '\n'
        format_str += '    Number of datapoints: {}\n'.format(self.__len__())
        format_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        format_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        format_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return format_str

IMG_FORMAT = ['.jpg', '.jpeg', '.png']

def data_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def dft_loader(path):
    return data_loader(path)

class ImgFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=dft_loader):
        super(ImgFolder, self).__init__(root, loader, IMG_FORMAT,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
