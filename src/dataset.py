import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class MVTecADDataset(Dataset):
    """
    Loads MVTec AD dataset for anomaly detection.
    """
    def __init__(self, root_dir, category, split="train", image_size=256):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        self.samples = []
        self._collect_files()
    def _collect_files(self):
        base = os.path.join(self.root_dir, self.category)
        if self.split == "train":
            good = os.path.join(base, "train", "good")
            for f in sorted(os.listdir(good)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(good, f), 0, "good"))
        elif self.split == "test":
            test_root = os.path.join(base, "test")
            for defect_type in sorted(os.listdir(test_root)):
                defect_folder = os.path.join(test_root, defect_type)
                if not os.path.isdir(defect_folder):
                    continue
                label = 0 if defect_type == "good" else 1
                for f in sorted(os.listdir(defect_folder)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self.samples.append(
                            (os.path.join(defect_folder, f), label, defect_type)
                        )
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label, defect_type = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label, defect_type, path
if __name__ == "__main__":
    ds = MVTecADDataset(
        root_dir="data/mvtec_anomaly_detection",
        category="bottle",
        split="train"
    )
    print("Train count:", len(ds))
    x, y, t, p = ds[0]
    print("Shape:", x.shape)
    print("Label:", y)
    print("Type:", t)
    print("Path:", p)



