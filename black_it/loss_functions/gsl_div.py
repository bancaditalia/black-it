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

"""
This module contains the GSL-div implementation.

- Lamperti, F. (2018).
  An information theoretic criterion for empirical validation of simulation models.
  Econometrics and Statistics, 5, 83-106.

Algorithm description:

0. We have 2 TS:
    a. {X} = 'observed', 1 realisation
    b. {y} = simulated, many realisations
1. Prepare the TS (discretize) {X} = {x1,x2,...,xt,...,xT}:
    1. take {xMax,xMin}
    2. partition in 'b' intervals - simbolize the TS
        1. each 'xt' becomes the bucket # that is assigned to 'xt_'
    3. the symbolized TS {x_} is subdivided in 'words' of len 'l'
        2. use a set of len(s) for 'l' eg {1,2,...,L}
    4. for each word len 'l' compute the vector of word occurrencies
        3. hence get 'L' such frequencies vectors: {x_l}
2. Compute the GSL-div:
    1. for each word len 'l' compute:
        1. Shannon entropy of Sx = {x_l}
        2. Shannon entropy of Sm = ({x_l} + {y_l})/2,
            where {y} is another TS. Ie we obs {X} and simulate {y}.
        3. compute weights to be given to each word len 'l': {w}
            1. use 'additively progressive weights'
            2. w(0) = 0
            3. w(l+1) = w(l) + 2/(L*(L-1))
        4. GSL-div = SUMMATION(1,...,L)w*(2Sm - Sx)
                    = SUMMATION(1,...,L)w*(2(avg entropy) - obs entropy)
3. Ensable runs (k simulations of the model)
    1. use a simple average for freq vectors {x_l}


**Note**: b,L don't increase much the comp power required (ie from (2,2) to (19,19) +20% time).

"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from black_it.loss_functions.base import BaseLoss
from black_it.utils.base import assert_

EPS = 0.00001  # np.finfo(float).eps


class GslDivLoss(BaseLoss):
    """
    Class for the Gsl-div loss.

    Example:
        >>> expected_loss = 0.39737637181336855
        >>> np.random.seed(11)
        >>> series1 = np.random.normal(0, 1, (100, 3))
        >>> series2 = np.random.normal(0, 1, (100, 3))
        >>> loss_func = GslDivLoss()
        >>> loss = loss_func.compute_loss(series1[None, :, :], series2)
        >>> assert np.isclose(expected_loss, loss)
    """

    def __init__(
        self,
        nb_values: int = None,
        nb_word_lengths: int = None,
        coordinate_weights: Optional[NDArray] = None,
    ) -> None:
        """
        Initialize the GSL-div loss object.

        Args:
            nb_values: number of values the digitised series can take
            nb_word_lengths: the number of word length to consider
            coordinate_weights: the weights of the loss coordinates
        """
        super().__init__(coordinate_weights)
        self.nb_values = nb_values
        self.nb_word_lengths = nb_word_lengths

    def compute_loss_1d(
        self, sim_data_ensemble: NDArray[np.float64], real_data: NDArray[np.float64]
    ) -> float:
        """
        Return the GSL-div measure.

        From (Lamperti, 2017):

        > The information loss about the behaviour of the stochastic process
          due to the symbolization becomes smaller and smaller as b increases.
          On the other side, low values of b would likely wash away
          processes’ noise the modeller might not be interested in.

        Args:
            sim_data_ensemble: the ensemble of simulated data
            real_data: the real data

        Returns:
            the GSL loss
        """
        N = len(real_data)
        ensemble_size = sim_data_ensemble.shape[0]

        if self.nb_values is None:
            nb_values = int((N - 1) / 2.0)
        else:
            nb_values = self.nb_values

        if self.nb_word_lengths is None:
            nb_word_lengths = int((N - 1) / 2.0)
        else:
            nb_word_lengths = self.nb_word_lengths

        # discretize real time series
        obs_xd = self.discretize(
            real_data,
            nb_values,
            np.min(real_data),
            np.max(real_data),
        )

        gsl_loss = 0.0

        # average loss over the ensemble
        for sim_data in sim_data_ensemble:

            # discretize simulated series
            sim_xd = self.discretize(
                sim_data, nb_values, np.min(sim_data), np.max(sim_data)
            )

            loss = self.gsl_div_1d_1_sample(
                sim_xd, obs_xd, nb_word_lengths, nb_values, N
            )

            gsl_loss += loss

        return gsl_loss / ensemble_size

    def gsl_div_1d_1_sample(
        self,
        sim_xd: NDArray,
        obs_xd: NDArray,
        nb_word_lengths: int,
        nb_values: int,
        N: int,
    ) -> float:
        """Compute the GSL-div for a single realisation of the simulated data.

        Args:
            sim_xd: discretised simulated series
            obs_xd: discretised real series
            nb_word_lengths: the number of word length to consider
            nb_values: number of values the digitised series can take
            N: the length of real and simulated series

        Returns:
            the computed loss
        """
        # outcome measure
        gsl_div = 0.0

        # weight
        weight = 0.0

        # for any word len:
        for word_length in range(1, nb_word_lengths + 1):
            sim_xw = self.get_words(sim_xd, word_length)
            obs_xw = self.get_words(obs_xd, word_length)
            m_xw = np.concatenate((sim_xw, obs_xw))
            sim_xp = self.get_words_est_prob(sim_xw)
            m_xp = self.get_words_est_prob(m_xw)
            base = float(nb_values**word_length)
            sim_entr = self.get_sh_entr(sim_xp, base)
            m_entr = self.get_sh_entr(m_xp, base)

            # update weight
            weight = weight + 2 / (nb_word_lengths * (nb_word_lengths + 1))

            # correction
            corr = ((len(m_xp) - 1) - (len(sim_xp) - 1)) / (2 * N)

            # add to measure
            gsl_divl = 2 * m_entr - sim_entr + corr
            gsl_div = gsl_div + weight * gsl_divl

        # end of cycle, return
        return gsl_div

    @staticmethod
    def discretize(
        time_series: NDArray[np.float64],
        nb_values: int,
        start_index: float,
        stop_index: float,
    ) -> NDArray[np.float64]:
        """
        Discretize the TS in 'nb_values' finite states.

        >>> GslDivLoss.discretize(
        ...     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     nb_values=3,
        ...     start_index=1,
        ...     stop_index=10
        ... )
        array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3])

        Args:
            time_series: any univariate time series
            nb_values: int, number of values the digitised series can take.
                It must be greater than 0.
            start_index: the starting point
            stop_index: the stopping point

        Returns:
            the discretised time series
        """
        linspace = np.linspace(start_index - EPS, stop_index + EPS, nb_values + 1)

        return np.searchsorted(linspace, time_series, side="left")

    @staticmethod
    def get_words(time_series: NDArray[np.float64], length: int) -> NDArray:
        """
        Return an overlapping array of words (int32) of 'length' given a discretised vector.

        >>> GslDivLoss.get_words(np.asarray([1, 2, 2, 2]), 2)
        array([12, 22, 22])

        Args:
            time_series: any univariate discretised time series
            length: int, len of words to be returned. Must be such that
                (len(time_series) + 1 - length) is positive.

        Returns:
            the time series of overlapping words

        """
        tswlen = len(time_series) + 1 - length
        assert_(tswlen >= 0, "the chosen word length is too high", exc_cls=ValueError)
        tsw = np.zeros(shape=(tswlen,), dtype=np.int32)

        for i in range(length):
            k = 10 ** (length - i - 1)
            tsw = tsw + time_series[i : tswlen + i] * k

        return tsw

    @staticmethod
    def get_words_est_prob(time_series: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Return an array of estimated probabilities given an array of words (int32).

        Args:
            time_series: any univariate array of words

        Returns:
            estimate of probabilities
        """
        _, count = np.unique(time_series, return_counts=True)
        est_p = np.divide(count, np.sum(count))
        return est_p

    @staticmethod
    def get_sh_entr(probs: NDArray[np.float64], log_base: float) -> float:
        """
        Return the Shannon entropy given an array of probabilities.

        Args:
            probs: an array of probabilities describing the discrete probability distribution
            log_base: the entropy logarithm base.

        Returns:
            the entropy of the discrete probability distribution
        """
        log = np.log(probs) / np.log(log_base)
        return -np.sum(np.multiply(probs, log))
