# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2022 Banca d'Italia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Implementation of the particle swarm sampler."""
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import assert_, digitize_data, positive_float


class ParticleSwarmSampler(
    BaseSampler
):  # pylint: disable=(too-many-instance-attributes)
    """
    Implementation of a particle swarm sampler.

    This sampler implements the particle swarm sampling method commonly used in particle swarm optimization (PSO),
    introduced in:

        Eberhart, Russell, and James Kennedy. "A new optimizer using particle swarm theory."
          MHS'95. Proceedings of the sixth international symposium on micro machine and human science. IEEE, 1995.

    In a particle swarm optimizer, there is a set of particles that are "evolved" by cooperation and competition
    among the individuals themselves through generations. Each particle adjusts its flying  according to its own
    flying experience and its companions’ flying experience. Each particle, in fact, represents a potential solution
    to a problem. Each particle is treated as a point in a D-dimensional space.  The ith particle is represented as
    Xi = (x_{i1},...,x_{iD}). The best previous position (the position giving the best fitness value) of any particle
    is recorded and represented as Pi = (p_{i1},...,p_{iD}). The index of the best particle among all the particles
    in the population is represented by the symbol g. The rate of the position change (velocity) for particle i is
    represented as Vi = (v_{i1},...,v_{iD}). The particles are manipulated according to the following equation:

        v_{id} = (ω * v_{id}) + (c1 * r1 * (p_{id} - x_{id})) + (c2 * r2 * (p_{gd} - x_{id})
        x_{id} = x_{id} + v_{id}

    Where:
        - ω is the inertia weight  to control the influence of the previous velocity;
        - c1 and c2 are positive values that represent the acceleration constants;
        - r1 and r2 are two random numbers uniformly distributed in the range of (0, 1).

    Note that p_{gd}, the global best position found across the dynamics, can optionally be computed by also
    considering the sampling performed by other samplers in order to let them interfere constructively with the
    Particle Swarm Sampler.
    """

    def __init__(
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        inertia: float = 0.9,
        c1: float = 0.1,
        c2: float = 0.1,
        global_minimum_across_samplers: bool = False,
    ) -> None:
        """
        Initialize the sampler.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            inertia: the inertia of the particles' motion
            c1: first acceleration constant
            c2: second acceleration constant
            global_minimum_across_samplers: if True, the global minimum attractor of the particles' dynamics is computed
                taking into consideration also parameters sampled by other samplers, default is False
        """
        # max_duplication_passes must be zero because the sampler is stateful
        super().__init__(
            batch_size, random_state=random_state, max_duplication_passes=0
        )

        # The batch size is the number of sampled parameters per iteration. In a Black-it sampler, each call to
        # sample_batch represent an iteration of the particle swarm sampler, so it seems natural to set the number of
        # particles to the batch size, as at each iteration sample_batch returns the current positions of the
        # particles.
        self.nb_particles = batch_size

        self._inertia = positive_float(inertia)
        self._c1 = positive_float(c1)
        self._c2 = positive_float(c2)
        self._global_minimum_across_samplers = global_minimum_across_samplers

        # all current particle positions; shape=(nb_particles, space dimensions)
        self._curr_particle_positions: Optional[NDArray] = None
        # all current particle velocities; shape=(nb_particles, space dimensions)
        self._curr_particle_velocities: Optional[NDArray] = None
        # best particle positions, i.e. ; shape=(nb_particles, space dimensions)
        self._best_particle_positions: Optional[NDArray] = None
        # losses of the best positions
        self._best_position_losses: Optional[NDArray] = None
        # particle id of the global best particle position
        self._global_best_particle_id: Optional[int] = None

        # best point in parameter space - could be the best across samplers
        self._best_point: Optional[NDArray] = None

        self._previous_batch_index_start: Optional[int] = None

    @property
    def is_set_up(self) -> bool:
        """Return true iff the sampler is already set up."""
        return self._curr_particle_positions is not None

    @property
    def inertia(self) -> float:
        """Get the inertia weight."""
        return self._inertia

    @property
    def c1(self) -> float:
        """Get the c1 constant."""
        return self._c1

    @property
    def c2(self) -> float:
        """Get the c2 constant."""
        return self._c2

    def _set_up(self, dims: int) -> None:
        """Set up the sampler."""
        self._curr_particle_positions = self.random_generator.random(
            size=(self.batch_size, dims)
        )
        self._curr_particle_velocities = (
            self.random_generator.random(
                size=cast(NDArray, self._curr_particle_positions).shape
            )
            - 0.5
        )
        self._best_particle_positions = self._curr_particle_positions
        # set losses to inf as we are interested to the min
        self._best_position_losses = np.full(self.nb_particles, np.inf)
        # we don't know yet which is the best index - initialize to 0
        self._global_best_particle_id = 0

    def _get_best_position(self) -> NDArray[np.float64]:
        """
        Get the position corresponding to the global optimum the particles should converge to.

        If _global_minimum_across_samplers is False, then this method returns the current position
        of the particle that in its history has sampled, so far, the best set of parameters.

        Else, if _global_minimum_across_samplers is True, then this method returns the point
        in parameter space that achieved the minimum loss. Note that this point could have been
        sampled by a different sampler than "self".

        Returns:
            a Numpy array
        """
        if not self._global_minimum_across_samplers:
            best_particle_positions = cast(NDArray, self._best_particle_positions)
            return best_particle_positions[self._global_best_particle_id]
        return cast(NDArray, self._best_point)

    def reset(self) -> None:
        """Reset the sampler."""
        self._curr_particle_positions = None
        self._curr_particle_velocities = None
        self._best_particle_positions = None
        self._best_position_losses = None
        self._global_best_particle_id = None
        self._previous_batch_index_start = None
        assert_(
            not self.is_set_up,
            message="reset call did not work, sampler still set up",
            exc_cls=RuntimeError,
        )

    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample a batch of parameters."""
        if not self.is_set_up:
            self._set_up(search_space.dims)
            self._previous_batch_index_start = len(existing_points)
            return digitize_data(
                cast(NDArray[np.float64], self._best_particle_positions),
                search_space.param_grid,
            )

        self._update_best(existing_points, existing_losses)
        self._do_step()

        p_bounds: NDArray[np.float64] = search_space.parameters_bounds
        sampled_points = p_bounds[0] + self._curr_particle_positions * (
            p_bounds[1] - p_bounds[0]
        )
        self._previous_batch_index_start = len(existing_points)

        return digitize_data(sampled_points, search_space.param_grid)

    def _update_best(
        self, existing_points: NDArray[np.float64], existing_losses: NDArray[np.float64]
    ) -> None:
        """Update the best local and global positions."""
        assert_(
            self._previous_batch_index_start is not None,
            exc_cls=AssertionError,
            message="should have been set",
        )

        # set best loss and best point
        best_point_index = np.argmin(existing_losses)
        self._best_point = existing_points[best_point_index]

        # set best particle position
        batch_index_start = cast(int, self._previous_batch_index_start)
        batch_index_stop = batch_index_start + self.batch_size
        previous_points = existing_points[batch_index_start:batch_index_stop]
        previous_losses = existing_losses[batch_index_start:batch_index_stop]
        for particle_id, (point, loss) in enumerate(
            zip(previous_points, previous_losses)
        ):
            best_particle_positions = cast(NDArray, self._best_particle_positions)
            best_position_losses = cast(NDArray, self._best_position_losses)
            if best_position_losses[particle_id] > loss:
                best_particle_positions[particle_id] = point
                best_position_losses[particle_id] = loss

                # check if also the global best should be updated
                best_global_loss = best_position_losses[self._global_best_particle_id]
                if loss < best_global_loss:
                    self._global_best_particle_id = particle_id

    def _do_step(self) -> None:
        """Do a step by updating particle positions and velocities."""
        curr_particle_positions = cast(NDArray, self._curr_particle_positions)
        curr_particle_velocities = cast(NDArray, self._curr_particle_velocities)
        best_particle_positions = cast(NDArray, self._best_particle_positions)
        r1_vec = self.random_generator.random(size=curr_particle_positions.shape)
        r2_vec = self.random_generator.random(size=curr_particle_positions.shape)
        new_particle_velocities = (
            self.inertia * curr_particle_velocities
            + self.c1
            * r1_vec
            * (best_particle_positions - self._curr_particle_positions)
            + self.c2
            * r2_vec
            * (self._get_best_position() - self._curr_particle_positions)  # type: ignore
        )

        self._curr_particle_positions = np.clip(
            self._curr_particle_positions + new_particle_velocities,
            a_min=0.0,
            a_max=1.0,
        )
        self._curr_particle_velocities = new_particle_velocities
