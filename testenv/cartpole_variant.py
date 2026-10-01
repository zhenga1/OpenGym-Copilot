from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import register, registry
from gymnasium.spaces import Box


class LooseCartPoleEnv(CartPoleEnv):
    """
    CartPole variant with looser failure conditions.

    The default CartPole-v1 terminates around +/-12 degrees and +/-2.4 cart position.
    This variant defaults to:
    - +/-30 degree pole angle
    - +/-4.8 cart position
    """

    def __init__(
        self,
        *args,
        x_threshold: float = 10,
        theta_threshold_degrees: float = 30.0,
        pole_length: float = 1.0,
        cart_width: float = 90.0,
        cart_height: float = 48.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.x_threshold = float(x_threshold)
        self.theta_threshold_radians = math.radians(float(theta_threshold_degrees))
        # Gymnasium CartPole stores half the physical pole length on `self.length`.
        # The default is 0.5, so 1.0 makes the rendered pole about 2x longer.
        self.length = float(pole_length)
        self.cart_width = float(cart_width)
        self.cart_height = float(cart_height)

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as exc:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from exc

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = self.cart_width
        cartheight = self.cart_height

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0
        carty = 100
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


def register_test_envs() -> None:
    env_id = "CartPoleLoose-v0"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="testenv.cartpole_variant:LooseCartPoleEnv",
            max_episode_steps=500,
        )


__all__ = ["LooseCartPoleEnv", "register_test_envs"]
