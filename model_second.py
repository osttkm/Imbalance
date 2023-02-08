import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_feature, out_future, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        self.in_feature = in_feature
        self.out_feature = out_future

        self.conv = nn.Conv2d(in_feature, out_future, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_future)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = self.bn(x) if self.use_bn else x  # bn層を使わない場合はxを代入
        x = self.relu(x)
        return x

class Classifier(nn.Module):
    def __init__(self,enc_dim,num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.fc_dim = enc_dim

        self.encoder = nn.Sequential(
            EncoderBlock(3      , enc_dim),
        )


        self.fc = nn.Sequential(
            # nn.Linear(self.fc_dim, 256),
            # nn.ReLU(),
            nn.Linear(self.fc_dim, self.num_classes),
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = F.adaptive_avg_pool2d(out,output_size=(1,1))
        # import pdb;pdb.set_trace()
        out = out.view(-1, self.fc_dim)
        out = self.fc(out)
        return out
