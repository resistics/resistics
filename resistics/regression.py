"""The regression module provides functions and classes for the following:

- Preparing gathered data for regression
- Performing the regression to calculate the components of the transfer function

Resistics has a few built in solvers, but makes it possible to define custom
solvers as required
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from loguru import logger
from regressioninc import LeastSquares

from resistics.common import (
    CancellationCallback,
    History,
    Metadata,
    ProcessingCancelled,
    ProcessingProgressCallback,
    ProcessingProgressEvent,
    ProcessingProgressState,
    ResisticsData,
    ResisticsProcess,
    WriteableMetadata,
)
from resistics.gather import GatheredData, SiteCombinedMetadata
from resistics.spectra import SpectraData, SpectraMetadata
from resistics.transfunc import Component, TransferFunction, get_component_key


class _RegressionResult(Protocol):
    # Minimal regressioninc result boundary required by the linear solver.
    @property
    def coefficients(self) -> np.ndarray: ...


class _FittableRegressor(Protocol):
    # Minimal regressioninc/plugin boundary required by the linear solver.
    @property
    def result_(self) -> _RegressionResult: ...

    def fit(self, predictors: np.ndarray, observations: np.ndarray, /) -> object: ...


@dataclass
class _ProgressReporter:
    task: str
    total: int
    progress_callback: ProcessingProgressCallback | None = None
    cancellation_callback: CancellationCallback | None = None
    current: int = 0

    def start(self, message: str) -> None:
        self._emit(ProcessingProgressState.started, message)

    def check_cancelled(self) -> None:
        if self.cancellation_callback is None or not self.cancellation_callback():
            return
        message = f"Cancelled {self.task.replace('_', ' ')}"
        self._emit(ProcessingProgressState.cancelled, message)
        raise ProcessingCancelled(message)

    def advance(self, message: str) -> None:
        self.current += 1
        self._emit(ProcessingProgressState.advanced, message)

    def complete(self, message: str) -> None:
        self._emit(ProcessingProgressState.completed, message)

    def fail(self, error: Exception) -> None:
        self._emit(
            ProcessingProgressState.failed,
            f"Failed {self.task.replace('_', ' ')}",
            error=str(error),
        )

    def _emit(
        self,
        state: ProcessingProgressState,
        message: str,
        error: str | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            ProcessingProgressEvent(
                state=state,
                task=self.task,
                current=self.current,
                total=self.total,
                message=message,
                error=error,
            )
        )


def _cross_channels(tf: TransferFunction) -> list[str]:
    return tf.cross_chans


def _dimensions(tf: TransferFunction) -> tuple[int, int]:
    return tf.n_out, tf.n_in


def _observations(
    tf: TransferFunction, out_powers: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        out_chan: out_powers[:, idx, ...].flatten()
        for idx, out_chan in enumerate(tf.out_chans)
    }


def _predictors(in_powers: np.ndarray) -> np.ndarray:
    values = np.swapaxes(in_powers, 1, 2)
    return values.reshape(-1, values.shape[-1])


def get_least_squares_regressor() -> _FittableRegressor:
    """Return the regressioninc least-squares regressor.

    :return: Fresh complex least-squares estimator.
    """
    return LeastSquares()


class RegressionInputMetadata(Metadata):
    """Metadata for regression input data, mainly to track processing history"""

    contributors: dict[str, SiteCombinedMetadata | SpectraMetadata]
    """Details about the data contributing to the regression input data"""
    history: History = History()
    """The processing history"""


class RegressionInputData(ResisticsData):
    """Class to hold data that will be input into a solver

    The purpose of regression input data is to provision for many different
    solvers and user written solvers.

    The regression input data has the following key attributes:

    - freqs
    - obs
    - preds

    The freqs attribute is a 1-D array of evaluation frequencies.

    The obs attribute is a dictionary of dictionaries. The parent dictionary has
    a key of the evaluation frequency index. The secondary dictionary has key of
    output channel. The values in the secondary dictionary are the observations
    for that output channel and have 1-D size:

    [n_wins x n_cross_chans].

    The preds attribute is a single level dictionary with key of evaluation
    frequency index and value of the predictors for the evaluation frequency.
    The predictors have 2-D shape:

    [n_wins x n_cross_chans, n_input_channels].


    For an example, consider the impedance tensor. This has:

    - output channels Ex, Ey
    - input channels Hx, Hy

    Call the cross channels

    - cross channels C1, C2

    For single site processing, the cross channels are often Hx and Hy, though
    this does not have to be the case and the source of this data changes for
    remote reference processing.

    In this case, the observations and predictors for the output channel Ex are:

    Observations Ex

    - win1 C1: <Ex_win1, conj(C1_win1)>
    - win1 C2: <Ex_win1, conj(C2_win1)>
    - win2 C1: <Ex_win2, conj(C1_win2)>
    - win2 C2: <Ex_win2, conj(C2_win1)>
    - ...

    Predictors Ex

    - win1 C1:  Zxx <Hx_win1, conj(C1_win1)>   Zxy <Hy_win1, conj(C1_win1)>
    - win1 C2:  Zxx <Hx_win1, conj(C2_win1)>   Zxy <Hy_win1, conj(C2_win1)>
    - win2 C1:  Zxx <Hx_win2, conj(C1_win2)>   Zxy <Hy_win2, conj(C1_win2)>
    - win2 C2:  Zxx <Hx_win2, conj(C2_win1)>   Zxy <Hy_win2, conj(C2_win1)>
    - ...

    Note that the predictors are the same regardless of the output channel,
    only the observations change.

    :param metadata: Contributor metadata and combined processing history.
    :param tf: Transfer-function channel definition to solve.
    :param freqs: Evaluation frequencies in hertz.
    :param obs: Per-frequency observations keyed by output channel.
    :param preds: Per-frequency predictor matrices.
    """

    def __init__(
        self,
        metadata: RegressionInputMetadata,
        tf: TransferFunction,
        freqs: list[float],
        obs: list[dict[str, np.ndarray]],
        preds: list[np.ndarray],
    ) -> None:
        self.metadata = metadata
        self.tf = tf
        self.freqs = freqs
        self.obs = obs
        self.preds = preds

    @property
    def n_freqs(self) -> int:
        """Get the number of frequencies"""
        return len(self.freqs)

    def get_inputs(self, freq_idx: int, out_chan: str) -> tuple[np.ndarray, np.ndarray]:
        """Get observations and predictions

        :param freq_idx: The evaluation frequency index
        :param out_chan: The output channel

        :return: Observations and predictons
        """
        return self.obs[freq_idx][out_chan], self.preds[freq_idx]


class ImpedanceTensorSetup(ResisticsProcess):
    """Create the standard MT impedance transfer-function definition."""

    output_type: ClassVar[str] = "transfer_function"

    def run(self) -> TransferFunction:
        """Return the default impedance tensor channel definition.

        :return: Electric outputs over magnetic inputs for standard MT processing.
        """
        from resistics.transfunc import ImpedanceTensor

        return ImpedanceTensor()


class RegressionPreparerGathered(ResisticsProcess):
    """Regression preparer for gathered data

    In nearly all cases, this is the regresson preparer to use. As input, it
    requires GatheredData.
    """

    input_types: ClassVar[dict[str, str]] = {
        "tf": "transfer_function",
        "gathered_data": "gathered_data",
    }
    output_type: ClassVar[str] = "regression_input"
    include_in_default_parameters: ClassVar[bool] = True

    def run(  # noqa: DOC105 - pydoclint cannot resolve callback type aliases
        self,
        tf: TransferFunction,
        gathered_data: GatheredData,
        *,
        progress_callback: ProcessingProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> RegressionInputData:
        """Create the RegressionInputData

        :param tf: The transfer function
        :param gathered_data: The gathered data
        :param progress_callback: Consumer for structured frequency progress.
        :param cancellation_callback: Callback checked before each evaluation frequency.

        :return: Data that can be used as input into a solver
        """
        logger.info("Preparing regression data")
        logger.info(f"Out chans site: {gathered_data.out_data.metadata.site_name}")
        logger.info(f"Out chans: {gathered_data.out_data.metadata.chans}")
        logger.info(f"In chans site: {gathered_data.in_data.metadata.site_name}")
        logger.info(f"In chans: {gathered_data.in_data.metadata.chans}")
        logger.info(f"Cross chans site: {gathered_data.cross_data.metadata.site_name}")
        logger.info(f"Cross chans: {gathered_data.cross_data.metadata.chans}")
        return self._get_regression_data(
            tf,
            gathered_data,
            progress_callback=progress_callback,
            cancellation_callback=cancellation_callback,
        )

    def _get_regression_data(  # noqa: DOC105 - callback aliases are documented
        self,
        tf: TransferFunction,
        gathered_data: GatheredData,
        *,
        progress_callback: ProcessingProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> RegressionInputData:
        """Get the regression input data

        :param tf: The transfer function
        :param gathered_data: The gathered data
        :param progress_callback: Consumer for structured frequency progress.
        :param cancellation_callback: Callback checked before each evaluation frequency.

        :return: Data to be used as input to a solver

        :raises ProcessingCancelled: If cancellation is requested before a frequency is prepared.
        :raises Exception: If frequency preparation fails.
        """
        freqs = []
        obs = []
        preds = []
        metadata = gathered_data.out_data.metadata
        logger.info(f"Preparing regression data for {metadata.n_evals} frequencies")
        reporter = _ProgressReporter(
            task="prepare_regression",
            total=metadata.n_evals,
            progress_callback=progress_callback,
            cancellation_callback=cancellation_callback,
        )
        reporter.start("Preparing regression frequencies")
        try:
            for idx, freq in enumerate(metadata.eval_freqs):
                reporter.check_cancelled()
                out_powers, in_powers = self._get_cross_powers(tf, gathered_data, idx)
                freqs.append(freq)
                obs_freq = self._get_obs(tf, out_powers)
                preds_freq = self._get_preds(tf, in_powers)
                obs.append(obs_freq)
                preds.append(preds_freq)
                reporter.advance(
                    f"Prepared regression frequency {idx + 1} of {metadata.n_evals}"
                )
        except ProcessingCancelled:
            raise
        except Exception as exc:
            reporter.fail(exc)
            raise
        reporter.complete(f"Prepared {metadata.n_evals} regression frequencies")
        record = self._get_record(
            f"Produced regression input data for {metadata.n_evals} frequencies"
        )
        metadata = RegressionInputMetadata(
            contributors={
                "out_data": gathered_data.out_data.metadata,
                "in_data": gathered_data.in_data.metadata,
                "cross_data": gathered_data.cross_data.metadata,
            }
        )
        metadata.history.add_record(record)
        return RegressionInputData(metadata, tf, freqs, obs, preds)

    def _get_cross_powers(
        self, tf: TransferFunction, gathered_data: GatheredData, eval_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get cross powers

        Gathered data for an evaluation frequency is:

        [n_wins, n_chans]

        To multiply each in/out channel with the cross channels, broadcasting is
        used. Using output channels as an example, this is what we have:

        out_data = [n_wins, n_out_chans]
        cross_data = [n_wins, n_cross_chans]

        The aim is to achieve an array that looks like this:

        cross_powers = [n_wins, n_out_chans, n_cross_chans]

        This can be achieved by numpy broadcasting the two arrays as follows

        out_data = [n_wins, n_out_chans, new_axis]
        cross_data = [n_wins, new_axis, n_cross_chans]

        :param tf: Definition of transfer function
        :param gathered_data: All the gathered data
        :param eval_idx: The evaluation frequency index

        :return: Cross powers with output channels and cross powers with input channels
        """
        # calculate the cross powers
        out_data = gathered_data.out_data.data[eval_idx]
        in_data = gathered_data.in_data.data[eval_idx]
        cross_data = gathered_data.cross_data.data[eval_idx]
        cross_data = np.conjugate(cross_data[:, np.newaxis, :])

        # multiply using broadcasting
        out_powers = out_data[..., np.newaxis] * cross_data
        in_powers = in_data[..., np.newaxis] * cross_data
        return out_powers, in_powers

    def _get_obs(
        self, tf: TransferFunction, out_powers: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Get observations for an output channel

        This is a single dimension array with shape

        [n_wins * n_cross_chans]

        :param tf: Definition of transfer function
        :param out_powers: The cross powers for the output channels

        :return: Dictionary with output channel as key and observations as value
        """
        return _observations(tf, out_powers)

    def _get_preds(self, tf: TransferFunction, in_powers: np.ndarray) -> np.ndarray:
        """Construct the predictors

        The in_powers is received with shape

        [n_wins, n_in_chans, n_cross_chans]

        The aim is to make this into

        [n_wins * n_cross_chans, n_in_chans]

        :param tf: Transfer function definition
        :param in_powers: The cross powers for the input channels

        :return: The predictors
        """
        return _predictors(in_powers)


class RegressionPreparerSpectra(ResisticsProcess):
    """Prepare regression data directly from spectra data

    This can be useful for running a single measurement

    **See Also**

    RegressionPreparerGathered : Produce regression input data from gathered
    data

    **Attributes**

    - **input_types** — Flow ports for a transfer function and spectra data.
    - **output_type** — Flow type produced for solver input.
    - **include_in_default_parameters** — Whether default parameter sets include this preparer.
    """

    input_types: ClassVar[dict[str, str]] = {
        "tf": "transfer_function",
        "spec_data": "spectra_data",
    }
    output_type: ClassVar[str] = "regression_input"
    include_in_default_parameters: ClassVar[bool] = True

    def run(  # noqa: DOC105 - pydoclint cannot resolve callback type aliases
        self,
        tf: TransferFunction,
        spec_data: SpectraData,
        *,
        progress_callback: ProcessingProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> RegressionInputData:
        """Construct regression input while emitting frequency progress.

        :param tf: Transfer-function definition.
        :param spec_data: Spectra to prepare for regression.
        :param progress_callback: Consumer for structured frequency progress.
        :param cancellation_callback: Callback checked before each evaluation frequency.

        :return: Prepared observations and predictors.

        :raises ProcessingCancelled: If cancellation is requested before a frequency is prepared.
        :raises Exception: If frequency preparation fails.
        """
        freqs = []
        obs = []
        preds = []
        total = sum(len(level.freqs) for level in spec_data.metadata.levels_metadata)
        reporter = _ProgressReporter(
            task="prepare_regression",
            total=total,
            progress_callback=progress_callback,
            cancellation_callback=cancellation_callback,
        )
        reporter.start("Preparing regression frequencies")
        try:
            for ilevel in range(spec_data.metadata.n_levels):
                level_metadata = spec_data.metadata.levels_metadata[ilevel]
                out_powers, in_powers = self._get_cross_powers(tf, spec_data, ilevel)
                for idx, freq in enumerate(level_metadata.freqs):
                    reporter.check_cancelled()
                    logger.info(
                        "Preparing regression data: "
                        f"level {ilevel}, freq. {idx} = {freq}"
                    )
                    freqs.append(freq)
                    obs.append(_observations(tf, out_powers[..., idx]))
                    preds.append(_predictors(in_powers[..., idx]))
                    reporter.advance(
                        f"Prepared regression frequency {reporter.current + 1} "
                        f"of {total}"
                    )
        except ProcessingCancelled:
            raise
        except Exception as exc:
            reporter.fail(exc)
            raise
        reporter.complete(f"Prepared {total} regression frequencies")
        record = self._get_record("Produced regression input data for spectra data")
        metadata = RegressionInputMetadata(contributors={"data": spec_data.metadata})
        metadata.history.add_record(record)
        return RegressionInputData(metadata, tf, freqs, obs, preds)

    def _get_cross_powers(
        self, tf: TransferFunction, spec_data: SpectraData, level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get cross powers

        Spectra data is:

        [n_wins, n_chans, n_freqs]

        To multiply each in/out channel with the cross channels, broadcasting is
        used. Using output channels as an example, this is what we have:

        out_data = [n_wins, n_out_chans, n_freqs]
        cross_data = [n_wins, n_cross_chans, n_freqs]

        The aim is to achieve an array that looks like this:

        cross_powers = [n_wins, n_out_chans, n_cross_chans, n_freqs]

        :param tf: Definition of transfer function
        :param spec_data: Spectra data for a decimation level
        :param level: The decimation level

        :return: Cross powers with output channels and cross powers with input channels
        """
        # prepare to calculate the crosspowers
        out_data = spec_data.get_chans(level, tf.out_chans)
        in_data = spec_data.get_chans(level, tf.in_chans)
        cross_data = spec_data.get_chans(level, _cross_channels(tf))
        cross_data = np.conj(cross_data[:, np.newaxis, ...])

        # multiply using broadcasting
        out_powers = out_data[..., np.newaxis, :] * cross_data
        in_powers = in_data[..., np.newaxis, :] * cross_data
        return out_powers, in_powers


class Solution(WriteableMetadata):
    """Class to hold a transfer function solution

    **Examples**

    ```{doctest}
    >>> from resistics.testing import solution_mt
    >>> solution = solution_mt()
    >>> print(solution.tf.to_string())
    | ex | = | ex_hx ex_hy | | hx |
    | ey |   | ey_hx ey_hy | | hy |
    >>> solution.n_freqs
    6
    >>> solution.freqs
    [100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
    >>> solution.periods.tolist()
    [0.01, 0.0125, 0.016666666666666666, 0.025, 0.05, 0.1]
    >>> solution.components["exhx"]
    Component(real=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], imag=[5.0, 5.0, 4.0, 4.0, 3.0, 3.0])
    >>> solution.components["exhy"]
    Component(real=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], imag=[-5.0, -4.0, -3.0, -2.0, -1.0, 1.0])

    ```

    To get the components as an array, either get_component or subscripting
    be used

    ```{doctest}
    >>> solution["exhy"]
    array([1.-5.j, 2.-4.j, 3.-3.j, 4.-2.j, 5.-1.j, 6.+1.j])
    >>> solution["ab"]
    Traceback (most recent call last):
    ...
    ValueError: Component ab not found in ['exhx', 'exhy', 'eyhx', 'eyhy']

    ```

    It is also possible to get the tensor values at a particular evaluation
    frequency

    ```{doctest}
    >>> solution.get_tensor(2)
    array([[ 2.+4.j,  3.-3.j],
           [-3.+3.j, -2.-4.j]])

    ```
    """

    tf: TransferFunction
    """The transfer function that was solved"""
    freqs: list[float]
    """The evaluation frequencies"""
    components: dict[str, Component]
    """The solution"""
    history: History
    """The processing history"""
    contributors: dict[str, SiteCombinedMetadata | SpectraMetadata]
    """The contributors to the solution with their respective details"""

    def __getitem__(self, key: str) -> np.ndarray:
        """Solution for a single component for all evaluation frequencies

        The arguments should be output channel followed by input channel

        :param key: The component key

        :return: The component values as an array

        :raises ValueError: If incorrect number of arguments
        """
        if not isinstance(key, str):
            raise ValueError("Subscripting takes only 1 argument != {len(arg)}")
        return self.get_component(key)

    @property
    def n_freqs(self):
        """Get the number of evaluation frequencies"""
        return len(self.freqs)

    @property
    def periods(self) -> np.ndarray:
        """Get the periods"""
        return np.reciprocal(self.freqs)

    def get_component(self, key: str) -> np.ndarray:
        """Get the solution for a single component for all the evaluation
        frequencies

        :param key: The component key

        :return: The component data in an array

        :raises ValueError: If the component does not exist in the solution
        """
        if key not in self.components:
            raise ValueError(
                f"Component {key} not found in {list(self.components.keys())}"
            )
        return self.components[key].to_numpy()

    def get_tensor(self, eval_idx: int) -> np.ndarray:
        """Get the tensor at a single evaluation frequency. This has shape:

        n_out_chans x n_in_chans

        :param eval_idx: The index of the evaluation frequency

        :return: The tensor as a numpy array
        """
        n_out, n_in = _dimensions(self.tf)
        tensor = np.zeros(shape=(n_out, n_in), dtype=np.complex128)
        for out_idx, out_chan in enumerate(self.tf.out_chans):
            for in_idx, in_chan in enumerate(self.tf.in_chans):
                key = get_component_key(out_chan, in_chan)
                tensor[out_idx, in_idx] = self.components[key].get_value(eval_idx)
        return tensor

    def to_dataframe(self) -> pd.DataFrame:
        """Get the solution as a dataframe.

        :return: Complex transfer-function components indexed by frequency.
        """
        soln_data = {comp: self.get_component(comp) for comp in self.components}
        index = self.freqs
        return pd.DataFrame(data=soln_data, index=index)


class Solver(ResisticsProcess):
    """General resistics solver"""

    input_types: ClassVar[dict[str, str]] = {"regression_input": "regression_input"}
    output_type: ClassVar[str] = "transfer_function"
    include_in_default_parameters: ClassVar[bool] = False

    def run(  # noqa: DOC105 - pydoclint cannot resolve callback type aliases
        self,
        regression_input: RegressionInputData,
        *,
        progress_callback: ProcessingProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> Solution:
        """Solve regression input with optional progress and cancellation.

        :param regression_input: Prepared regression observations and predictors.
        :param progress_callback: Consumer for structured frequency progress.
        :param cancellation_callback: Callback checked before each evaluation frequency.

        :return: Transfer-function solution.

        :raises NotImplementedError: Always; concrete solvers must implement this method.
        """
        del regression_input, progress_callback, cancellation_callback
        raise NotImplementedError("Run not implemented in parent Solver class")


class SolverLinear(Solver):
    """Base class for linear solvers"""

    fit_intercept: bool = False
    """Flag for adding an intercept term"""

    def _solve(  # noqa: DOC105 - callback aliases are documented
        self,
        regression_input: RegressionInputData,
        model: _FittableRegressor,
        *,
        progress_callback: ProcessingProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> Solution:
        """Get the regression solution for all evaluation frequencies

        :param regression_input: The regression input data
        :param model: The model to use to solve the linear regressions
        :param progress_callback: Consumer for structured frequency progress.
        :param cancellation_callback: Callback checked before each evaluation frequency.

        :return: The solution for the transfer function

        :raises ProcessingCancelled: If cancellation is requested before a frequency is solved.
        :raises Exception: If fitting a frequency fails.
        """
        n_freqs = regression_input.n_freqs
        tf = regression_input.tf
        n_out, n_in = _dimensions(tf)
        tensors = np.ndarray((n_freqs, n_out, n_in), dtype=np.complex128)
        logger.info(f"Solving for {n_freqs} evaluation frequencies")
        reporter = _ProgressReporter(
            task="solve_regression",
            total=n_freqs,
            progress_callback=progress_callback,
            cancellation_callback=cancellation_callback,
        )
        reporter.start("Solving regression frequencies")
        try:
            for eval_idx in range(n_freqs):
                reporter.check_cancelled()
                for iout, out_chan in enumerate(tf.out_chans):
                    obs, preds = regression_input.get_inputs(eval_idx, out_chan)
                    tensors[eval_idx, iout] = self._get_coef(model, obs, preds)
                reporter.advance(
                    f"Solved regression frequency {eval_idx + 1} of {n_freqs}"
                )
        except ProcessingCancelled:
            raise
        except Exception as exc:
            reporter.fail(exc)
            raise
        reporter.complete(f"Solved {n_freqs} regression frequencies")
        return self._get_solution(tf, regression_input, tensors)

    def _get_coef(
        self, model: _FittableRegressor, obs: np.ndarray, preds: np.ndarray
    ) -> np.ndarray:
        """Get coefficients for a single evaluation frequency and output channel

        :param model: RegressionInC-compatible estimator.
        :param obs: The observations
        :param preds: The predictors

        :return: The coefficients

        :raises ValueError: If the regressor completes without producing coefficients.
        """
        model.fit(preds, obs)
        try:
            coefficients = model.result_.coefficients
        except AttributeError as exc:
            raise ValueError("Regressor did not produce coefficients") from exc
        if coefficients is None:
            raise ValueError("Regressor did not produce coefficients")
        return coefficients

    def _get_solution(
        self,
        tf: TransferFunction,
        regression_input: RegressionInputData,
        tensors: np.ndarray,
    ) -> Solution:
        """Get the solution

        :param tf: The transfer function
        :param regression_input: The regression input data
        :param tensors: The coefficients

        :return: The transfer function solution
        """
        components = {}
        for out_idx, out_chan in enumerate(tf.out_chans):
            for in_idx, in_chan in enumerate(tf.in_chans):
                key = get_component_key(out_chan, in_chan)
                values = tensors[:, out_idx, in_idx]
                components[key] = Component(
                    real=values.real.tolist(), imag=values.imag.tolist()
                )
        history = History(**regression_input.metadata.history.model_dump())
        message = f"Solved {len(regression_input.freqs)} evaluation frequencies"
        history.add_record(self._get_record(message))
        return Solution(
            tf=tf,
            freqs=regression_input.freqs,
            components=components,
            history=history,
            contributors=regression_input.metadata.contributors,
        )


class SolverOLS(SolverLinear):
    """Solve each evaluation frequency with ordinary least squares."""

    include_in_default_parameters: ClassVar[bool] = True
    n_jobs: int = -2
    """Number of jobs to run"""

    def run(  # noqa: DOC105 - pydoclint cannot resolve callback type aliases
        self,
        regression_input: RegressionInputData,
        *,
        progress_callback: ProcessingProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> Solution:
        """Run ordinary least squares regression with structured progress.

        :param regression_input: Prepared regression observations and predictors.
        :param progress_callback: Consumer for structured frequency progress.
        :param cancellation_callback: Callback checked before each evaluation frequency.

        :return: Transfer-function solution.
        """
        model = get_least_squares_regressor()
        return self._solve(
            regression_input,
            model,
            progress_callback=progress_callback,
            cancellation_callback=cancellation_callback,
        )


class SolutionWriter(ResisticsProcess):
    """Write a transfer-function solution to a job's staging output."""

    input_types: ClassVar[dict[str, str]] = {"solution": "transfer_function"}
    output_type: ClassVar[str] = "job_result"
    runtime_requirements: ClassVar[list[str]] = ["staging_output_path"]

    def execute(
        self, inputs: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, str]:
        """Write the supplied solution to ``solution.json``.

        :param inputs: Flow inputs containing the transfer-function solution.
        :param context: Runtime context containing the staging output path.

        :return: Result path exposed to the job runner.
        """
        path = Path(context["staging_output_path"])
        path.mkdir(parents=True, exist_ok=False)
        inputs["solution"].write(path / "solution.json")
        return {"result_path": str(path)}
