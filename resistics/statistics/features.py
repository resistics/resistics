from typing import Union, List, Dict, Sequence
import numpy as np

from resistics.common.base import ResisticsBase
from resistics.spectra.data import SpectrumData, PowerData


class FeatureInput(ResisticsBase):
    def __init__(self, data: ResisticsBase, evalfreqs: np.ndarray):
        self.data: ResisticsBase = data
        self.evalfreqs = evalfreqs
        self.process()

    def process(self):
        raise NotImplementedError("Parent class does not have a process method")


class FeatureInputSpectrum(FeatureInput):
    def __init__(
        self,
        data: SpectrumData,
        evalfreqs: np.ndarray,
        winLen: int = 9,
        winType: str = "hann",
    ):
        self.crosspowers: Union[PowerData, None] = None
        self.evalpowers: Union[PowerData, None] = None
        self.winLen: int = winLen
        self.winType: str = winType
        super().__init__(data, evalfreqs)

    def process(self):
        from resistics.spectra.calculator import crosspowers

        self.crosspowers = crosspowers(self.data)
        self.crosspowers.smooth(self.winLen, self.winType, inplace=True)
        self.evalpowers = self.crosspowers.interpolate(self.evalfreqs)


# class MultiWindow(self, data: List[ResisticsBase])


class Feature(ResisticsBase):
    def __init__(self, name: str):
        self._name: str = name

    @property
    def name(self):
        return self._name

    def prepare_output(self, evalfreqs: np.ndarray) -> Dict[float, Dict[str, float]]:
        output = {}
        for efreq in evalfreqs:
            output[efreq] = {}
        return output

    def evaluate(self, data) -> Dict[float, Dict[str, float]]:
        raise NotImplementedError("Every child class needs to have an evaluate method")


class PowerSpectralDensity(Feature):
    def __init__(self, channels: Union[List[str], None] = None):
        super().__init__("power_spectral_density")
        self._channels = channels
        if self._channels is None:
            self._channels = ["Ex", "Ey", "Hx", "Hy"]
            self.printWarning(f"Using default channels {self._channels}")

    def feature_label(self, chan: str) -> str:
        return f"psd{chan}"

    def evaluate(self, data: FeatureInputSpectrum) -> Dict[float, Dict[str, float]]:
        """Calculate power spectral density

        This is calculated out as power spectrum divided by the duration
        """
        powers: PowerData = data.evalpowers
        duration = data.data.duration
        output = self.prepare_output(data.evalfreqs)
        for chan in self._channels:
            key = self.feature_label(chan)
            autopowers = powers[chan, chan] / duration
            for eidx, efreq in enumerate(data.evalfreqs):
                output[efreq][key] = autopowers[eidx]
        return output


class CrosspowerAbsolute(Feature):
    def __init__(self):
        super().__init__("crosspower_absolute")

    def feature_label(self, chan1: str, chan2: str) -> str:
        return f"xabs{chan1}{chan2}"

    def evaluate(self, data: FeatureInputSpectrum) -> Dict[float, Dict[str, float]]:
        """Calculate absolute value"""
        powers: PowerData = data.evalpowers
        output = self.prepare_output(data.evalfreqs)
        for chan1 in powers.primaryChans:
            for chan2 in powers.secondaryChans:
                key = self.feature_label(chan1, chan2)
                absval = np.absolute(powers[chan1, chan2])
                for eidx, efreq in enumerate(data.evalfreqs):
                    output[efreq][key] = absval[eidx]
        return output


class Coherence(Feature):
    def __init__(self, coh_pairs: Union[List[str], None] = None):
        super().__init__("coherence")
        self._coh_pairs = coh_pairs
        if self._coh_pairs is None:
            self._coh_pairs = [["Ex", "Hy"], ["Ey", "Hx"]]
            self.printWarning(f"Using default statistic features {self._coh_pairs}")

    def feature_label(self, chan1: str, chan2: str) -> str:
        return f"coh{chan1}{chan2}"

    def evaluate(self, data: FeatureInputSpectrum) -> Dict[float, Dict[str, float]]:
        powers: PowerData = data.evalpowers
        output = self.prepare_output(data.evalfreqs)
        for pair in self._coh_pairs:
            chan1 = pair[0]
            chan2 = pair[1]
            key = self.feature_label(chan1, chan2)
            coherence = powers.getCoherence(chan1, chan2)
            for eidx, efreq in enumerate(data.evalfreqs):
                output[efreq][key] = coherence[eidx]
        return output


class PolarisationDirection(Feature):
    def __init__(self, pol_pairs: Union[List[str], None] = None):
        super().__init__("polarisation_direction")
        self._pol_pairs = pol_pairs
        if self._pol_pairs is None:
            self._pol_pairs = [["Ex", "Ey"], ["Hx", "Hy"]]
            self.printWarning(f"Using default statistic features {self._pol_pairs}")

    def feature_label(self, chan1: str, chan2: str) -> str:
        return f"pol{chan1}{chan2}"

    def evaluate(self, data: FeatureInputSpectrum) -> Dict[float, Dict[str, float]]:
        powers: PowerData = data.evalpowers
        output = self.prepare_output(data.evalfreqs)
        for pair in self._pol_pairs:
            chan1 = pair[0]
            chan2 = pair[1]
            key = self.feature_label(chan1, chan2)
            # calculate polarisation directions
            nom = 2 * powers[chan1, chan2].real
            denom = powers[chan1, chan1] - powers[chan2, chan2]
            poldirs = (np.arctan(nom / denom) * (180.0 / np.pi)).real
            for eidx, efreq in enumerate(data.evalfreqs):
                output[efreq][key] = poldirs[eidx]
        return output


class TransferFunction(Feature):
    def __init__(
        self,
        in_chans: Union[List[str], None] = None,
        out_chans: Union[List[str], None] = None,
        cross_chans: Union[List[str], None] = None,
    ):
        super().__init__("transfer_function")
        self._in_chans = in_chans
        self._out_chans = out_chans
        self._cross_chans = cross_chans
        if self._in_chans is None:
            self._in_chans = ["Hx", "Hy"]
            self.printWarning(f"Using default input channels {self._in_chans}")
        if self._out_chans is None:
            self._out_chans = ["Ex", "Ey"]
            self.printWarning(f"Using default output channels {self._out_chans}")
        if self._cross_chans is None:
            self._cross_chans = self._in_chans + self._out_chans
            self.printWarning(f"Using default cross channels {self._cross_chans}")
        self._in_size = len(self._in_chans)
        self._out_size = len(self._out_chans)
        self._cross_size = len(self._cross_chans)
        # intercept option
        self._intercept = False
        # prepare arrays
        self._obs = np.empty(shape=(self._out_size, self._cross_size), dtype="complex")
        self._reg = np.empty(
            shape=(self._out_size, self._cross_size, self._in_size), dtype="complex"
        )

    def feature_label(self, chan1: str, chan2: str, component: str) -> str:
        return f"tf{chan1}{chan2}{component}"

    def evaluate(self, data: FeatureInputSpectrum) -> Dict[float, Dict[str, float]]:
        from resistics.regression.robust import olsModel

        powers: PowerData = data.evalpowers
        output = self.prepare_output(data.evalfreqs)
        for eidx, efreq in enumerate(data.evalfreqs):
            # prepare lstsq equation
            for iout, out_chan in enumerate(self._out_chans):
                for icross, cross_chan in enumerate(self._cross_chans):
                    # this is the observation row where, iout is the observed output
                    self._obs[iout, icross] = powers[out_chan, cross_chan, eidx]
                    for iin, in_chan in enumerate(self._in_chans):
                        self._reg[iout, icross, iin] = powers[in_chan, cross_chan, eidx]
            # solve
            for iout, out_chan in enumerate(self._out_chans):
                observation = self._obs[iout, :]
                predictors = self._reg[iout, :, :]
                soln = olsModel(predictors, observation, intercept=self._intercept)
                params = soln.params[1:] if self._intercept else soln.params
                for iin, in_chan in enumerate(self._in_chans):
                    key_real = self.feature_label(out_chan, in_chan, "Real")
                    key_imag = self.feature_label(out_chan, in_chan, "Imag")
                    output[efreq][key_real] = params[iin].real
                    output[efreq][key_imag] = params[iin].imag
        return output


class TransferFunctionMT(Feature):
    """Caculate out MT focussed features including resistivity and phase and partial coherence"""

    def __init__(
        self,
        in_chans: Union[List[str], None] = None,
        out_chans: Union[List[str], None] = None,
        cross_chans: Union[List[str], None] = None,
        tipper: bool = False,
    ):
        super().__init__(in_chans, out_chans, cross_chans)
        self.tipper = tipper
        # set the name again
        self._name = "transfer_function_mt"

    def feature_label_res(self, chan1: str, chan2: str) -> str:
        return f"res{chan1}{chan2}"

    def feature_label_phase(self, chan1: str, chan2: str) -> str:
        return f"phs{chan1}{chan2}"

    def feature_label_bivar(self, chan1: str) -> str:
        return f"bivar{chan1}"

    def feature_label_partial(self, chan1: str, chan2: str) -> str:
        return f"par{chan1}{chan2}"

    def evaluate(self, data: FeatureInputSpectrum) -> Dict[float, Dict[str, float]]:
        output = super().evaluate(data)
        tf_elements = {}
        for efreq in output:
            tf_elements[efreq] = {}
            for out_chan in self._out_chans:
                for in_chan in self._in_chans:
                    key_real = self.feature_label(out_chan, in_chan, "Real")
                    z_out_in_real = output[efreq][key_real]
                    key_imag = self.feature_label(out_chan, in_chan, "Imag")
                    z_out_in_imag = output[efreq][key_imag]
                    key = out_chan + in_chan
                    tf_elements[efreq][key] = z_out_in_real + z_out_in_imag * 1j
        output = self._evaluate_res_phase(data, tf_elements, output)
        output = self._evaluate_bivariate_coherence(data, tf_elements, output)
        return self._evaluate_partial_coherence(data, output)

    def _evaluate_res_phase(
        self,
        data: FeatureInputSpectrum,
        tf_elements: Dict[str, complex],
        output: Dict[float, Dict[str, float]],
    ) -> Dict[float, Dict[str, float]]:
        for eidx, efreq in enumerate(data.evalfreqs):
            for out_chan in self._out_chans:
                for in_chan in self._in_chans:
                    z_out_in = tf_elements[efreq][out_chan + in_chan]
                    res = (0.2 * np.power(np.absolute(z_out_in), 2)) / efreq
                    phase = np.angle(z_out_in, deg=True)
                    output[efreq][self.feature_label_res(out_chan, in_chan)] = res
                    output[efreq][self.feature_label_phase(out_chan, in_chan)] = phase
        return output

    def _evaluate_bivariate_coherence(
        self,
        data: FeatureInputSpectrum,
        tf_elements: Dict[str, complex],
        output: Dict[float, Dict[str, float]],
    ) -> Dict[float, Dict[str, float]]:
        powers: PowerData = data.evalpowers
        for eidx, efreq in enumerate(data.evalfreqs):
            for out_chan in self._out_chans:
                bivar = 0
                for in_chan in self._in_chans:
                    z_out_in = tf_elements[efreq][out_chan + in_chan]
                    bivar += z_out_in * powers[in_chan, out_chan, eidx]
                bivar = np.absolute(bivar) / powers[out_chan, out_chan, eidx].real
                output[efreq][self.feature_label_bivar(out_chan)] = bivar
        return output


    def _get_coherences(self, powers: PowerData, out_chan: str) -> Dict[str, float]:
        coherences = {}
        for in_chan in self._in_chans:
            coherences[in_chan] = powers.getCoherence(out_chan, in_chan)
        return coherences
    
    def _get_coherence_sum(self, coherences: Dict[str, float], in_chans: Sequence[str]) -> List[str]:
        coherence_sum = 0
        for in_chan in in_chans:
            coherence_sum += coherences[in_chan]
        return coherence_sum


    def _evaluate_partial_coherence(
        self,
        data: FeatureInputSpectrum,
        output: Dict[float, Dict[str, float]],
    ) -> Dict[float, Dict[str, float]]:
        powers: PowerData = data.evalpowers
        for eidx, efreq in enumerate(data.evalfreqs):
            for out_chan in self._out_chans:
                bivar = output[efreq][self.feature_label_bivar(out_chan)]
                # get coherences
                coherences = self._get_coherences(powers, out_chan)
                for in_chan in self._in_chans:
                    # want to calculate partial coherence out_chan, in_chan
                    # remove the influence of other in_chans
                    other_chans = set(self._in_chans) - set([in_chan])
                    coherence_sum = self._get_coherence_sum(coherences, other_chans)
                    nom = bivar - coherence_sum
                    denom = 1 - coherence_sum
                    key = self.feature_label_partial(out_chan, in_chan)
                    output[efreq][key] = nom/denom
        return output
