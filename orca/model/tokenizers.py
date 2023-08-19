import functools as ft
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from orca.model.clip import CLIPTextTokenizer, CLIPVisionTokenizer, clip_weights_loader
from orca.model.transformer import MlpBlock
from orca.model.vision import encoders

EPS = 1e-6


# adapted from https://github.com/google-research/robotics_transformer/blob/master/tokenizers/token_learner.py
class TokenLearner(nn.Module):
    num_tokens: int
    bottleneck_dim: int = 64
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        if len(inputs.shape) == 4:
            inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1])
        x = nn.LayerNorm()(inputs)
        x = MlpBlock(
            mlp_dim=self.bottleneck_dim,
            out_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
        )(x, train=train)
        x = jnp.transpose(x, (0, 2, 1))  # (batch, num_tokens, h*w)
        x = nn.softmax(x, axis=-1)
        return jnp.einsum("bna,baf->bnf", x, inputs)


# adapted from https://github.com/google-research/robotics_transformer/blob/master/tokenizers/image_tokenizer.py
class ImageTokenizer(nn.Module):
    encoder: str
    encoder_kwargs: dict = None
    use_token_learner: bool = False
    num_tokens: int = 8  # this is not enforced unless use_token_learner is True
    conditioning_type: str = "none"

    @nn.compact
    def __call__(
        self,
        observations,
        goals=None,
        train: bool = True,
    ):
        # observations["image"] is (batch, obs_horizon, height, width, channel)
        # goals["image"] is (batch, height, width, channel)
        b, t, h, w, c = observations["image"].shape
        if self.conditioning_type == "none":
            # late-fusion architecture, image encoder doesn't see task and obs together
            image = observations["image"]
            image = jnp.reshape(image, (b * t, h, w, c))
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "goal_image":
            # early-fusion goal-image only architecture, concatenate obs and goal image channel-wise
            image = jnp.concatenate(
                [observations["image"][:, -1], goals["image"]], axis=-1
            )
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "goal_image_no_obs":
            image = goals["image"]
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "film_language":
            # encode task and pass into encoder with FiLM
            image = observations["image"]
            image = jnp.reshape(image, (b * t, h, w, c))
            lang = goals["language"]
            lang = lang[:, None, :].repeat(t, axis=1)
            lang = jnp.reshape(lang, (b * t, -1))
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(
                image, cond_var=lang
            )
            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

        if self.use_token_learner:
            image_tokens = jnp.reshape(
                image_tokens, (b * t, -1, image_tokens.shape[-1])
            )
            image_tokens = TokenLearner(num_tokens=self.num_tokens)(
                image_tokens, train=train
            )
        return image_tokens


class LanguageTokenizer(nn.Module):
    encoder: str = None
    encoder_kwargs: dict = None
    num_tokens: int = 1

    @nn.compact
    def __call__(
        self,
        observations,
        goals=None,
        train: bool = True,
    ):
        # TODO (andre) will need an actual encoder if we want token-level embeddings

        # add a time dimension to language
        if goals["language"].ndim == 2:
            tokens = goals["language"][:, None, :]
        else:
            tokens = goals["language"]

        return tokens


class ActionTokenizer(nn.Module):
    action_dim: int
    vocab_size: int
    normalization_type: str = "bounds"
    low: float = 0
    high: float = 1

    def setup(self):
        if self.normalization_type == "bounds":
            self.thresholds = jnp.linspace(self.low, self.high, self.vocab_size + 1)
        elif self.normalization_type == "normal":
            self.thresholds = norm.ppf(jnp.linspace(EPS, 1 - EPS, self.vocab_size + 1))
        else:
            raise ValueError

    def __call__(self, actions, mode: str = "tokenize"):
        if mode == "tokenize":
            if self.normalization_type == "bounds":
                actions = jnp.clip(actions, self.low + EPS, self.high - EPS)
            actions = actions[..., None]
            token_one_hot = (actions < self.thresholds[1:]) & (
                actions >= self.thresholds[:-1]
            ).astype(jnp.uint8)
            action_tokens = jnp.argmax(token_one_hot, axis=-1)
            return action_tokens
        elif mode == "detokenize":
            action_tokens = actions
            one_hot = jax.nn.one_hot(action_tokens, self.vocab_size)
            bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
            actions = jnp.sum(one_hot * bin_avgs, axis=-1)
            return actions


tokenizers = {
    "obs-tokenizer": ft.partial(
        ImageTokenizer,
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
        ),
        conditioning_type="none",
        num_tokens=64,
    ),
    "sim-obs-tokenizer": ft.partial(
        ImageTokenizer,
        encoder="resnetv1-18-bridge",
        encoder_kwargs=dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
        ),
        conditioning_type="none",
        num_tokens=16,
    ),
    "goal-tokenizer": ft.partial(
        ImageTokenizer,
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
        ),
        conditioning_type="goal_image_no_obs",
        num_tokens=64,
    ),
    "goal-obs-tokenizer": ft.partial(
        ImageTokenizer,
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
        ),
        conditioning_type="goal_image",
        num_tokens=64,
    ),
    "sim-goal-obs-tokenizer": ft.partial(
        ImageTokenizer,
        encoder="resnetv1-18-bridge",
        encoder_kwargs=dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
        ),
        conditioning_type="goal_image",
        num_tokens=16,
    ),
    "obs-film-language-tokenizer": ft.partial(
        ImageTokenizer,
        encoder="resnetv1-34-bridge-film",
        encoder_kwargs=dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
        ),
        conditioning_type="film_language",
        num_tokens=64,
    ),
    "language-tokenizer": LanguageTokenizer,
    "clip-obs-tokenizer": ft.partial(
        CLIPVisionTokenizer,
        conditioning_type="obs_image",
    ),
    "clip-goal-tokenizer": ft.partial(
        CLIPVisionTokenizer,
        conditioning_type="goal_image",
    ),
    "clip-text-tokenizer": CLIPTextTokenizer,
    # TODO (andre) other possible tokenizers:
    # "language-wordpiece-tokenizer": use token-level embeddings
    # "proprio": use proprio from observations
}

# TODO this belongs somewhere else
weights_loaders = {
    "clip": clip_weights_loader,
}

if __name__ == "__main__":
    import jax
    import numpy as np

    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    action = np.broadcast_to(action, [2, 2, 7])
    tokenizer = ActionTokenizer(
        action_dim=7, vocab_size=256, normalization_type="normal"
    )
    params = tokenizer.init(jax.random.PRNGKey(0), action)
    action_tokens = tokenizer.apply(params, action)
    detokenized_actions = tokenizer.apply(params, action_tokens, mode="detokenize")

    print(action)
    print(action_tokens)
    print(detokenized_actions)