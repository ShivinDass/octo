import copy
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

# pylint: disable=W0223
class NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                'Non linear function {} is not supported'.format(non_linear))

    def forward(self, input_value):
        return self.module(input_value)

    def __repr__(self):
        return repr(self.module)

class MultiHeadedMLPModule(nn.Module):

    def __init__(self,
                 n_heads,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._layers = nn.ModuleList()

        output_dims = self._check_parameter_for_output_layer(
            'output_dims', output_dims, n_heads)
        output_w_inits = self._check_parameter_for_output_layer(
            'output_w_inits', output_w_inits, n_heads)
        output_b_inits = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits, n_heads)
        output_nonlinearities = self._check_parameter_for_output_layer(
            'output_nonlinearities', output_nonlinearities, n_heads)

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(prev_size, output_dims[i])
            output_w_inits[i](linear_layer.weight)
            output_b_inits[i](linear_layer.bias)
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module('non_linearity',
                                        NonLinearity(output_nonlinearities[i]))

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'n_heads ({}), but {} provided.')
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]

class GaussianMLPTwoHeadedModule(nn.Module):
    """Base of GaussianMLPModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

        self._shared_mean_log_std_network = MultiHeadedMLPModule(
            n_heads=2,
            input_dim=self._input_dim,
            output_dims=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=[
                nn.init.zeros_,
                lambda x: nn.init.constant_(x, self._init_std.item())
            ],
            layer_normalization=self._layer_normalization)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        self._min_std_param = buffers['min_std_param']
        self._max_std_param = buffers['max_std_param']

    def _get_mean_and_log_std(self, *inputs):
        return self._shared_mean_log_std_network(*inputs)
    
    def forward(self, *inputs):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist


class TanhNormal(torch.distributions.Distribution):
    r"""A distribution induced by applying a tanh transformation to a Gaussian random variable.

    Algorithms like SAC and Pearl use this transformed distribution.
    It can be thought of as a distribution of X where
        :math:`Y ~ \mathcal{N}(\mu, \sigma)`
        :math:`X = tanh(Y)`

    Args:
        loc (torch.Tensor): The mean of this distribution.
        scale (torch.Tensor): The stdev of this distribution.

    """ # noqa: 501

    def __init__(self, loc, scale):
        self._normal = Independent(Normal(loc, scale), 1)
        super().__init__(validate_args=False)

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
        """The log likelihood of a sample on the this Tanh Distribution.

        Args:
            value (torch.Tensor): The sample whose loglikelihood is being
                computed.
            pre_tanh_value (torch.Tensor): The value prior to having the tanh
                function applied to it but after it has been sampled from the
                normal distribution.
            epsilon (float): Regularization constant. Making this value larger
                makes the computation more stable but less precise.

        Note:
              when pre_tanh_value is None, an estimate is made of what the
              value is. This leads to a worse estimation of the log_prob.
              If the value being used is collected from functions like
              `sample` and `rsample`, one can instead use functions like
              `sample_return_pre_tanh_value` or
              `rsample_return_pre_tanh_value`


        Returns:
            torch.Tensor: The log likelihood of value on the distribution.

        """
        # pylint: disable=arguments-differ
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + epsilon + value) / (1 + epsilon - value)) / 2
        norm_lp = self._normal.log_prob(pre_tanh_value)
        ret = (norm_lp - torch.sum(
            torch.log(self._clip_but_pass_gradient((1. - value**2)) + epsilon),
            axis=-1))
        return ret

    def sample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients `do not` pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        z = self._normal.rsample(sample_shape)
        return torch.tanh(z)

    def rsample_with_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal distribution.

        Returns the sampled value before the tanh transform is applied and the
        sampled value with the tanh transform applied to it.

        Args:
            sample_shape (list): shape of the return.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Samples from this distribution.
            torch.Tensor: Samples from the underlying
                :obj:`torch.distributions.Normal` distribution, prior to being
                transformed with `tanh`.

        """
        z = self._normal.rsample(sample_shape)
        return z, torch.tanh(z)

    def cdf(self, value):
        """Returns the CDF at the value.

        Returns the cumulative density/mass function evaluated at
        `value` on the underlying normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.cdf(value)

    def icdf(self, value):
        """Returns the icdf function evaluated at `value`.

        Returns the icdf function evaluated at `value` on the underlying
        normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.icdf(value)

    @classmethod
    def _from_distribution(cls, new_normal):
        """Construct a new TanhNormal distribution from a normal distribution.

        Args:
            new_normal (Independent(Normal)): underlying normal dist for
                the new TanhNormal distribution.

        Returns:
            TanhNormal: A new distribution whose underlying normal dist
                is new_normal.

        """
        # pylint: disable=protected-access
        new = cls(torch.zeros(1), torch.zeros(1))
        new._normal = new_normal
        return new

    def expand(self, batch_shape, _instance=None):
        """Returns a new TanhNormal distribution.

        (or populates an existing instance provided by a derived class) with
        batch dimensions expanded to `batch_shape`. This method calls
        :class:`~torch.Tensor.expand` on the distribution's parameters. As
        such, this does not allocate new memory for the expanded distribution
        instance. Additionally, this does not repeat any args checking or
        parameter broadcasting in `__init__.py`, when an instance is first
        created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance(instance): new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            Instance: New distribution instance with batch dimensions expanded
            to `batch_size`.

        """
        new_normal = self._normal.expand(batch_shape, _instance)
        new = self._from_distribution(new_normal)
        return new

    def enumerate_support(self, expand=True):
        """Returns tensor containing all values supported by a discrete dist.

        The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Note:
            Calls the enumerate_support function of the underlying normal
            distribution.

        Returns:
            torch.Tensor: Tensor iterating over dimension 0.

        """
        return self._normal.enumerate_support(expand)

    @property
    def mean(self):
        """torch.Tensor: mean of the distribution."""
        return torch.tanh(self._normal.mean)

    @property
    def variance(self):
        """torch.Tensor: variance of the underlying normal distribution."""
        return self._normal.variance

    def entropy(self):
        """Returns entropy of the underlying normal distribution.

        Returns:
            torch.Tensor: entropy of the underlying normal distribution.

        """
        return self._normal.entropy()

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0., upper=1.):
        """Clipping function that allows for gradients to flow through.

        Args:
            x (torch.Tensor): value to be clipped
            lower (float): lower bound of clipping
            upper (float): upper bound of clipping

        Returns:
            torch.Tensor: x clipped between lower and upper.

        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = ((upper - x) * clip_up + (lower - x) * clip_low)
        return x + clip

    def __repr__(self):
        """Returns the parameterization of the distribution.

        Returns:
            str: The parameterization of the distribution and underlying
                distribution.

        """
        return self.__class__.__name__



class TanhGaussianBCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=None, *args, **kwargs):
        super().__init__()
        self.num_hidden_layers = len(hidden_sizes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._obs_dim = obs_dim
        self._action_dim = act_dim

        self._module = GaussianMLPTwoHeadedModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=nn.ReLU,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            output_nonlinearity=None,
            output_w_init=nn.init.xavier_uniform_,
            output_b_init=nn.init.zeros_,
            init_std=1.0,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
            std_parameterization='exp',
            layer_normalization=False,
            normal_distribution_cls=TanhNormal)

    def forward(self, obs):
        dist = self._module(obs)
        return dist

    def loss(self,
             pred_act_dist,
             action_batch,
             reduce_loss=True,
        ):
        log_prob = pred_act_dist.log_prob(action_batch)
        if reduce_loss:
            return -1 * log_prob.mean()
        else:
            return -1 * log_prob
    
    def get_action(self, ob):
        with torch.no_grad():
            ob = torch.tensor(ob[None], dtype=torch.float).to('cuda')
            dist = self._module(ob)
            act = dist.mean.cpu().detach().numpy()[0]
            return act

    def embed(self, obs):
        with torch.no_grad():
            x = obs
            for i, layer in enumerate(self._module._shared_mean_log_std_network._layers):
                if i + 1 == self.num_hidden_layers:
                    layer = layer[:-1]
                x = layer(x)
        return x
    

def create_mw_policy(seed):
    # if seed > 0:
    #     seed_everything(seed)

    obs_dim = 39
    policy = TanhGaussianBCPolicy(
        obs_dim=obs_dim,
        act_dim=4,
        hidden_sizes=[400, 400, 400],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.)
    ).float().cuda()
    return policy
