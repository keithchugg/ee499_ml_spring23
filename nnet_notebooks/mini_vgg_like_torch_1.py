# Define the model
class VggLikeBlock(nn.Module):
    def __init__(self, in_channels, num_filters, ksize=3, drate=0.25, pad='same'):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, num_filters, kernel_size=ksize, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, kernel_size=ksize, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=drate)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ClassifierMlp(nn.Module):
    def __init__(self, n_in, n_hidden, n_classes, drate=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout1d(p=drate),
            nn.Linear(n_hidden, n_classes)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MiniVgg(nn.Module):
    def __init__(self, b1_filters=32, b2_filters=64, H=28, fc_nodes=512, n_classes=10):
        super().__init__()
        self.block1 = VggLikeBlock(1, b1_filters)
        self.block2 = VggLikeBlock(b1_filters, b2_filters)
        assert H % 4 == 0, f'the image height and width must be a multiple of 4: you passed H = {H}'
        mlp_in_size = (H * H // 16) * b2_filters
        self.classifier = ClassifierMlp(mlp_in_size, fc_nodes, n_classes)
    def forward(self, x):
        batch_size = x.size(0)
        y = self.block1(x)
        y = self.block2(y)
        y = y.view(batch_size, -1)
        y = self.classifier(y)
        return y