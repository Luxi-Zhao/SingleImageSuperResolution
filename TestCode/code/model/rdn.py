# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C = nConvLayers

        self.head = RDB_Conv(G0, G)
        
        convs = []
        for c in range(1, C):
            convs.append(RDB_Conv(c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        '''
        # Unwrapping CM mechanism 
        conv1 = self.relu(self.conv(x)) # RDB_Conv(G0)
        cout1_dense = self.relu(torch.cat([x, conv1], 1)) # G0 + G
        
        conv2 = self.relu(self.conv2(conv1_dense)) # RDB_Conv(G0 + G)
        cout2_dense = self.relu(torch.cat([conv1_dense, conv2], 1))  # x conv1 conv2  G0 + 2G
        
        conv3 = self.relu(self.conv2(conv2_dense)) # RDB_Conv(G0 + 2G)
        cout3_dense = self.relu(torch.cat([conv2_dense, conv2], 1)) # x conv1 conv2 conv3
   
        # Without CM -------------------------------------------- 
        conv1 = self.relu(self.conv(x))
        
        conv2 = self.relu(self.conv2(conv1)) # RDB_Conv(G)
        cout2_dense = self.relu(torch.cat([conv1, conv2], 1))  # conv1 conv2 2G
        
        conv3 = self.relu(self.conv2(conv2_dense)) # RDB_Conv(2G)
        cout3_dense = self.relu(torch.cat([conv2_dense, conv2], 1))  # conv1 conv2 conv3
        
        # SRDenseNet ----------------------------------------------
        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))
        '''
        
        # New code -----------------------------------------------
        conv1 = self.head(x)
        return self.LFF(self.convs(conv1)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (20, 9, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return self.UPNet(x)