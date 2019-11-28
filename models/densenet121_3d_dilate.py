import torch.nn as nn
from torchvision import models
from lib.rpn_util import *
import torch


def dilate_layer(layer, val):

    layer.dilation = val
    layer.padding = val


class RPN(nn.Module):


    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()

        self.base = base

        # del self.base.transition3.pool

        # dilate
        # dilate_layer(self.base.denseblock4.denselayer1.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer2.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer3.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer4.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer5.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer6.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer7.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer8.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer9.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer10.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer11.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer12.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer13.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer14.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer15.conv2, 2)
        # dilate_layer(self.base.denseblock4.denselayer16.conv2, 2)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]

        # self.prop_feats = nn.Sequential(
        #     nn.Conv2d(self.base[-1].num_features, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        self.prop_feats = nn.Sequential(
            nn.Conv2d(self.base.layer3[-1].conv3.out_channels, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )


        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1)

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors
    

        # bbox estimation
        # self.dimension = nn.Sequential(
        #     nn.Conv2d(self.base.layer3[-1].conv3.out_channels, 512, 1),
        #     # nn.Linear(4 * self.feat_size[0] * self.feat_size[1], 512),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(512, 3 * self.num_anchors, 1),
        #     # IM TRULY SORRY
        #     # nn.Linear(512, (conf.batch_size * 3 * self.num_anchors * self.feat_size[0] * self.feat_size[1]) // 512),
        #     nn.LeakyReLU(0.1)
        # )
        # self.bin = 2
        # self.orientation = nn.Sequential(
        #     nn.Conv2d(self.base.h[-1].conv3.out_channels, 256, 1),
        #     nn.Linear(8 * self.feat_size[0] * self.feat_size[1], 256),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(256, self.num_anchors, 1),
        #     # IM TRULY SORRY
        #     nn.Linear(256, (conf.batch_size * self.num_anchors * self.feat_size[0] * self.feat_size[1]) // 256),
        #     nn.LeakyReLU(0.1)
        # )
        # self.confidence = nn.Sequential(
        #     nn.Conv2d(self.base.layer3[-1].conv3.out_channels, 256, 1),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(256, self.bin, 1),
        # )

        # bbox estimation




    def forward(self, x):

        batch_size = x.size(0)
        
        
        # self.base = base()

        # resnet
        # x = self.base(x)
        # x = self.base()
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)

        prop_feats = self.prop_feats(x)
        # print('props_feat', prop_feats.size())

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)
        # print('bbox_w3d', bbox_w3d.size())

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # bbox estimation
        # bbox_base_x = self.bbox_base.features(x)
        # bbox_base_x = torch.flatten(bbox_base_x, 1)

        bbox_base_x = torch.cat([bbox_x, bbox_y, bbox_w, bbox_h], dim=1)
        print('bbox_base_x', bbox_base_x.size())

        # dimension = self.dimension(bbox_base_x)  # .view(512, -1)).view(batch_size, self.num_anchors * 3, feat_h, feat_w)      
        # orientation = self.orientation(bbox_base_x)  # .view(256, -1)).view(batch_size, self.num_anchors, feat_h, feat_w)  
        # orientation = orientation.view(self.bin, -1)
        # L2 normalize, no need lamda
        # orientation = F.normalize(orientation, p=2, dim=1)
        # confidence = self.confidence(bbox_base_x)

        # update with predicted alpha, [-pi, pi]
        # alpha = recover_angle(self.anchors, confidence, self.bin)

        # compute global and local orientation
        # rot_global, rot_local = compute_orientaion(P2, xmax, xmin, alpha)

        # print('dimension', dimension.size())
        # print('orientation', orientation.size())
        # print('confidence', confidence.size())
        # bbox estimation


        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bbox_w3d = flatten_tensor(dimension[:,:self.num_anchors,:,:].view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        # bbox_h3d = flatten_tensor(dimension[:,self.num_anchors:self.num_anchors*2,:,:].view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        # bbox_l3d = flatten_tensor(dimension[:,self.num_anchors*2:self.num_anchors*3,:,:].view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        # bbox_rY3d = flatten_tensor(orientation.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # print('final bbox_x', bbox_x.size())
        # print('final bbox_h3d', bbox_h3d.size())
        # print('final bbox_l3d', bbox_l3d.size())
        # print('final bbox_rY3d', bbox_rY3d.size())


        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = self.rois.type(torch.cuda.FloatTensor)

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):

    train = phase.lower() == 'train'

    # densenet121 = models.densenet121(pretrained=train)
    densenet121 = models.resnet101(pretrained=train)

    # rpn_net = RPN(phase, densenet121.features, conf)
    rpn_net = RPN(phase, densenet121, conf)

    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net
