import torch
import numpy as np

class LinearPolicy(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(LinearPolicy, self).__init__()
        self.d_out = d_out
        self.linear = torch.nn.Linear(d_in, d_out)


    def forward(self, tx):
        u = self.linear(tx).reshape((1, self.d_out))
        return torch.ones((1, 1)), u

    def logParameters(self, writer, it):
        for param in list(self.named_parameters(prefix='LinearPolicy', recurse=True)):
            for scalar_it in range(len(param[1].data.view(-1))):
                writer.add_scalar(param[0] + "/" + str(scalar_it), param[1].data.view(-1)[scalar_it].item(), it)


class NonlinearPolicy(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(NonlinearPolicy, self).__init__()

        self.d_out  = d_out
        self.n_hidden = d_in * 2 * 2

        self.linear1 = torch.nn.Linear(d_in, self.n_hidden)
        self.activation1 = torch.tanh
        self.linear2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.activation2 = torch.tanh
        self.linear3 = torch.nn.Linear(self.n_hidden, self.d_out)

    def forward(self, tx):
        z_h1 = self.activation1(self.linear1(tx))
        u = self.linear3(z_h1).reshape((1, self.d_out))
        return torch.ones((1, 1)), u

    def logParameters(self, writer, it):
        for param in list(self.named_parameters(prefix='NonlinearPolicy', recurse=True)):
            for scalar_it in range(len(param[1].data.view(-1))):
                writer.add_scalar(param[0] + "/" + str(scalar_it), param[1].data.view(-1)[scalar_it].item(), it)


class TwoLayerNLP(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(TwoLayerNLP, self).__init__()

        self.d_out  = d_out
        self.n_hidden = 128

        self.linear1 = torch.nn.Linear(d_in, self.n_hidden)
        self.activation1 = torch.tanh
        self.linear2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.activation2 = torch.tanh
        self.linear3 = torch.nn.Linear(self.n_hidden, self.d_out)

    def forward(self, tx):
        z_h1 = self.activation1(self.linear1(tx))
        z_h2 = self.activation2(self.linear2(z_h1))
        u = self.linear3(z_h2).reshape((1, self.d_out))
        return torch.ones((1, 1)), u

    def logParameters(self, writer, it):
        for param in list(self.named_parameters(prefix='TwoLayerNLP', recurse=True)):
            for scalar_it in range(len(param[1].data.view(-1))):
                writer.add_scalar(param[0] + "/" + str(scalar_it), param[1].data.view(-1)[scalar_it].item(), it)


class ExpertMixturePolicy(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(ExpertMixturePolicy, self).__init__()

        self.num_experts = 8
        self.n_hidden = d_in * 4
        self.d_out = d_out

        self.linear1 = torch.nn.Linear(d_in, self.n_hidden)
        self.activation1 = torch.tanh

        self.selector_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.num_experts),
            # torch.nn.Softmax(dim=-1)
            torch.nn.Sigmoid()
        )

        self.expert_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, d_out*self.num_experts)
        )

    def forward(self, tx):
        z_h = self.activation1(self.linear1(tx))
        pi_nonNormalized = self.selector_net(z_h)
        pi = pi_nonNormalized / pi_nonNormalized.sum()

        u_experts = self.expert_net(z_h).reshape((self.num_experts, self.d_out))

        return pi, u_experts

    def logParameters(self, writer, it):
        for param in list(self.named_parameters(prefix='ExpertMixPolicy', recurse=True)):
            for scalar_it in range(len(param[1].data.view(-1))):
                writer.add_scalar(param[0]+"/"+str(scalar_it), param[1].data.view(-1)[scalar_it].item(), it)
