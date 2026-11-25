import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
import sys

class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb



class ClassifierBlock(Module):
    """Classifier block with a learned G transform."""
    def __init__(self, out_features, nfea, device):
        super(ClassifierBlock, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)
        self.G_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.G = nn.Linear(out_features, out_features).to(device)
        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)
        self.device = device

    def forward(self, input, lap, view):
        input1 = self.S(self.S_norm(input))
        input2 = self.U(self.U_norm(view))
        output = torch.mm(lap, input)
        output = self.G(self.G_norm(output))
        return input1 + input2 - output


class TrustworthyNet_classfier(nn.Module):
    """Semi-supervised classifier.

    ZZ_init is a plain list, so its linear layers are not in the optimizer or state_dict.
    """
    def __init__(self, nfeats, n_view, n_classes, n, args, device):
        super(TrustworthyNet_classfier, self).__init__()
        if n_view < 2 and args.fusion_type == 'trust':
            raise ValueError('DS fusion requires at least two views')
        self.n_classes = n_classes
        self.blocks = args.block
        self.block = args.block
        self.n_view = n_view
        self.n = n
        self.ZZ_init = []
        self.fusion_type = args.fusion_type
        self.input_type = args.input_type
        if self.input_type not in ('feature', 'similarity'):
            raise ValueError('input_type must be feature or similarity')
        if self.fusion_type not in ('trust', 'average', 'weight', 'attention'):
            raise ValueError('fusion_type must be trust, average, weight or attention')
        if self.fusion_type != 'trust':
            self.fusionlayer = FusionLayer(n_view, self.fusion_type, n_classes, hidden_size=64)

        dims = nfeats if self.input_type == 'feature' else [n] * n_view
        for i in range(n_view):
            setattr(self, f'block{i}', ClassifierBlock(n_classes, dims[i], device))
        for d in dims:
            self.ZZ_init.append(nn.Linear(d, n_classes).to(device))
        # Present but not used by forward; retain historical RNG/parameter order.
        self.Z_init = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n, n_classes, device=device)))
        self.theta = nn.Parameter(torch.FloatTensor([args.thre]), requires_grad=True).to(device)
        self.bn_input_01 = nn.BatchNorm1d(self.n_classes, momentum=0.5).to(device)
        self.device = device

    def DS_Combin(self, alpha):
        def DS_Combin_two(alpha1, alpha2):
            b, S, E, u = {}, {}, {}, {}
            for v, a in enumerate((alpha1, alpha2)):
                S[v] = torch.sum(a, dim=1, keepdim=True)
                E[v] = a - 1
                b[v] = E[v] / S[v].expand(E[v].shape)
                u[v] = self.n_classes / S[v]
            bb = torch.bmm(b[0].view(-1, self.n_classes, 1),
                           b[1].view(-1, 1, self.n_classes))
            bu = torch.mul(b[0], u[1].expand(b[0].shape))
            ub = torch.mul(b[1], u[0].expand(b[0].shape))
            C = torch.sum(bb, dim=(1, 2), out=None) - torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))
            S_a = self.n_classes / u_a
            return torch.mul(b_a, S_a.expand(b_a.shape)) + 1
        alpha_a = DS_Combin_two(alpha[0], alpha[1])
        for v in range(1, len(alpha) - 1):
            alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def self_active_l1(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def self_active_l21(self, x, thershold):
        nw = torch.norm(x)
        if nw > thershold:
            x = (nw - 1 / thershold) * x / nw
        else:
            x = torch.zeros_like(x)
        return x

    def forward(self, features, sim, lap, active='l1'):
        inputs = features if self.input_type == 'feature' else sim
        if inputs is None:
            raise ValueError('Missing feature or similarity inputs')
        out_tmp = 0
        for j in range(self.n_view):
            out_tmp += self.ZZ_init[j](inputs[j] / 1.0)
        output_z = [self.self_active_l1(out_tmp / len(inputs)).to(self.device)]
        for i in range(self.block):
            evidence = []
            for j in range(self.n_view):
                h = getattr(self, f'block{j}')(output_z[-1], lap[j], inputs[j] / 1.0)
                evidence.append(h)
            h = self.DS_Combin(evidence) if self.fusion_type == 'trust' else self.fusionlayer(evidence)
            h = self.bn_input_01(h)
            if active == 'l1':
                output_z.append(self.self_active_l1(h))
            elif active == 'l21':
                output_z.append(self.self_active_l21(h, self.theta))
            else:
                raise ValueError(f'Unsupported classifier activation: {active}')
        return output_z[1:]


# Preferred corrected spelling, with the existing misspelling kept for
# backward compatibility with semiClassifier.py and external scripts.
TrustworthyNet_classifier = TrustworthyNet_classfier


class ClusterBlock(Module):
    def __init__(self, out_features, nfea, device):
        super(ClusterBlock, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)
        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)
        self.device = device

    def forward(self, input, lap, view):
        input1 = self.S(self.S_norm(input))
        input2 = self.U(self.U_norm(view))
        output = torch.mm(lap, input)
        output = input1 + input2 - output
        return output


class TrustworthyNet_cluster(nn.Module):
    """Clustering network.

    fusion_type='trust' uses Dempster-Shafer fusion. average, weight, and
    attention use FusionLayer, registered after the other modules.
    """
    def __init__(self, nfeats, n_view, n_classes, n, Z_init, args, device):
        super(TrustworthyNet_cluster, self).__init__()
        self.n_classes = n_classes
        self.block = args.block
        self.n_view = n_view
        self.n = n
        self.fusion_type = args.fusion_type
        self.input_type = args.input_type
        if self.fusion_type not in ('trust', 'average', 'weight', 'attention'):
            raise ValueError('fusion_type must be trust, average, weight or attention')
        if self.fusion_type == 'trust' and n_view < 2:
            raise ValueError('DS fusion requires at least two views')
        if self.input_type == 'feature':
            dims = nfeats
        elif self.input_type == 'similarity':
            dims = [n] * n_view
        else:
            raise ValueError('input_type must be feature or similarity')

        # Keep the historical block0/block1/... names and initialization order.
        for i in range(n_view):
            setattr(self, f'block{i}', ClusterBlock(n_classes, dims[i], device))
        self.Z_init = torch.as_tensor(Z_init, dtype=torch.float32, device=device)
        self.theta = nn.Parameter(torch.FloatTensor([args.thre]), requires_grad=True).to(device)
        self.bn_input_01 = nn.BatchNorm1d(self.n_classes, momentum=0.6).to(device)
        if self.fusion_type != 'trust':
            self.fusionlayer = FusionLayer(n_view, self.fusion_type, n_classes, hidden_size=64)
        self.device = device

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.n_classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.n_classes, 1), b[1].view(-1, 1, self.n_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.n_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def self_active_l1(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def self_active_l21(self, x, thershold):
        nw = torch.norm(x)
        if nw > thershold:
            x = (nw - 1 / thershold) * x / nw
        else:
            x = torch.zeros_like(x)
        return x

    def forward(self, features, sim, lap, active='l1'):
        output_z = [self.Z_init]
        for i in range(self.block):
            evidence = []
            for j in range(self.n_view):
                view = features[j] if self.input_type == 'feature' else sim[j]
                h = getattr(self, f'block{j}')(output_z[-1], lap[j], view / 1.0)
                evidence.append(h)
            h = self.DS_Combin(evidence) if self.fusion_type == 'trust' else self.fusionlayer(evidence)
            h = self.bn_input_01(h)
            if active == 'l1':
                output_z.append(self.self_active_l1(h))
            elif active == 'l21':
                output_z.append(self.self_active_l21(h, self.theta))
            else:
                raise ValueError(f'Unsupported clustering activation: {active}')
        return output_z[1:]


# Compatibility alias: classification uses ClassifierBlock, not ClusterBlock.
Block = ClassifierBlock
