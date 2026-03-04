# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Evaluation metrics for continuous, categorical, and signature analysis.

The module provides:

- :class:`EvaluationMetric` for classic scalar and spatial skill scores;
- :class:`CategoricalEvaluationMetric` for event-based contingency metrics;
- :class:`SignatureEvaluationMetric` for flow-duration-curve based signatures.
"""

import numpy as np
from scipy.stats import pearsonr
from eofs.standard import Eof
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt


class EvaluationMetric:
    """Continuous-value evaluation metrics."""

    def __init__(self, sim, obs):
        """
        Initialize the metric class with simulated and observed values.

        Parameters
        ----------
        sim : array-like
            Simulated values.
        obs : array-like
            Observed values.
        """
        self.sim = np.array(sim)
        self.obs = np.array(obs)

    def MSE(self):
        """
        Computes the Mean Squared Error (MSE) between the simulated and observed values.

        Returns
        -------
        float
            The calculated MSE.
        """
        mse = sum((self.sim - self.obs) ** 2) / len(self.sim)
        return mse

    def RMSE(self):
        """
        Computes the Root Mean Squared Error (RMSE) between the simulated and observed values.

        Returns
        -------
        float
            The calculated RMSE.
        """
        rmse = (sum((self.sim - self.obs) ** 2) / len(self.sim)) ** 0.5
        return rmse

    def RRMSE(self):
        """
        Computes the Relative Root Mean Squared Error (RRMSE) between the simulated and observed values.

        Returns
        -------
        float
            The calculated RRMSE.
        """
        rrmse = (
            (sum((self.sim - self.obs) ** 2)) ** 0.5 / len(self.sim) / self.obs.mean()
        )
        return rrmse

    def R(self, confidence: float = 0.95):
        """
        Computes the Pearson correlation coefficient (R) and its significance.

        Parameters
        ----------
        confidence : float, optional
            The confidence level to determine the significance, by default 0.95.

        Returns
        -------
        tuple
            A tuple containing the correlation coefficient (r), p-value, and significance:
            - r : float
                The Pearson correlation coefficient.
            - p_value : float
                The p-value corresponding to the correlation coefficient.
            - significance : int
                A value indicating the significance of the correlation:
                1 for positive correlation, -1 for negative, and 0 for no significant correlation.
        """
        r, p_value = pearsonr(self.sim, self.obs)
        # or np.corrcoef(self.sim, self.obs)[0, 1]
        significance = 0
        if p_value < 1 - confidence:
            if r > 0:
                significance = 1
            elif r < 0:
                significance = -1

        return r, p_value, significance

    def R2(self):
        """
        Computes the R-squared (R2) value of the linear fit between the simulated and observed values.

        Returns
        -------
        float
            The calculated R2 value.
        """
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        r2 = r**2

        return r2

    def NSE(self):
        """
        Computes the Nash-Sutcliffe Efficiency (NSE) coefficient.

        The NSE measures how well the simulated values match the observed values,
        with higher values indicating better performance.

        Returns
        -------
        float
            The calculated NSE value.
        """
        nse = 1 - sum((self.obs - self.sim) ** 2) / sum(
            (self.obs - self.sim.mean()) ** 2
        )
        return nse

    def Bias(self):
        """
        Computes the bias between the simulated and observed values.

        The bias is the mean difference between the observed and simulated values.

        Returns
        -------
        float
            The calculated bias.
        """
        bias = (self.obs - self.sim).mean()
        return bias

    def PBias(self):
        """
        Computes the Percent Bias (PBias) between the simulated and observed values.

        Returns
        -------
        float
            The calculated PBias.
        """
        pbias = sum(self.obs - self.sim) / sum(self.obs) * 100
        return pbias

    def KGE(self, components=False):
        """
        Computes the Kling-Gupta Efficiency (KGE) metric between the simulated and observed values.

        The KGE metric is based on the correlation coefficient (r), the ratio of means (beta),
        and the ratio of standard deviations (gamma).

        Returns
        -------
        float
            The calculated KGE value.
        """
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        beta = np.mean(self.sim) / np.mean(self.obs)
        gamma = np.std(self.sim) / np.std(self.obs)

        kge = 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5
        
        if components:
            return kge, r, beta, gamma
        else:
            return kge

    def KGE_m(self, components=False):
        """
        Computes the modified Kling-Gupta Efficiency (KGE-m) metric between the simulated and observed values.

        The KGE-m metric is similar to KGE but adjusts the gamma term to account for
        the relative standard deviations of the observed and simulated values.

        Returns
        -------
        float
            The calculated KGE-m value.
        """
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        beta = np.mean(self.sim) / np.mean(self.obs)
        gamma = (np.std(self.sim) / np.mean(self.sim)) / (
            np.std(self.obs) / np.mean(self.obs)
        )

        kge = 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5
        
        if components:
            return kge, r, beta, gamma
        else:
            return kge
    
    def ESS(self, lats=None, n_modes=None, remove_mean=True, mask=None):
        """Compute EOF-based error similarity score for spatiotemporal fields.

        Parameters
        ----------
        lats : array-like, optional
            Latitude values used to build cosine-latitude weights.
        n_modes : int, optional
            Number of EOF modes used in the score.
        remove_mean : bool, optional
            Whether to remove the temporal mean before EOF decomposition.
        mask : ndarray of bool, optional
            Spatial mask of shape ``(nlat, nlon)``. Masked cells are excluded.

        Returns
        -------
        numpy.ndarray
            Score series with length ``ntime``.
        """
        assert self.obs.shape == self.sim.shape, "sim and obs must have identical dimensions"
        
        ntime, nlat, nlon = self.sim.shape
        
        # reshape
        sim_2d = self.sim.reshape(ntime, nlat * nlon)
        obs_2d = self.obs.reshape(ntime, nlat * nlon)
        
        # mask
        if mask is not None:
            assert mask.shape == (nlat, nlon), "mask shape must match spatial dimensions"
            flat_mask = mask.flatten()
            sim_2d = sim_2d[:, flat_mask]
            obs_2d = obs_2d[:, flat_mask]
            
        # combine sim and obs array
        combine_2d = np.concatenate([sim_2d, obs_2d], axis=0)
        
        # Compute anomalies by removing the time-mean
        if remove_mean:
            combine_2d -= np.mean(combine_2d, axis=0)
        
        # latitude weights are applied before the computation of EOFs
        if lats is not None:
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            wgts = np.sqrt(coslat)[..., np.newaxis]
            
            if mask is not None:
                # Also apply mask to weights
                wgts = wgts.flatten()[flat_mask]
        else:
            wgts = None
        
        # solver
        solver = Eof(combine_2d, weights=wgts)
        
        # get values
        pcs = solver.pcs(npcs=n_modes)
        eigvals = solver.eigenvalues()[:n_modes]
        varfrac = solver.varianceFraction(neigs=n_modes)
        
        loadings = pcs * np.sqrt(eigvals)  # shape: (2*ntime, n_modes)
        
        # get ess
        obs_loadings = loadings[:ntime]        # shape: (ntime, n_modes)
        sim_loadings = loadings[ntime:]  # shape: (ntime, n_modes)
        
        diffs = np.abs(obs_loadings - sim_loadings)  # shape: (n_days, n_modes)
        
        ess = np.sum(diffs * varfrac[np.newaxis, :], axis=1)  # shape: (ntime,)
        
        return ess

    def spatialPCC(self, mask=None):
        """Compute spatial Pearson correlation for each time step.

        Parameters
        ----------
        mask : ndarray of bool, optional
            Spatial mask of shape ``(nlat, nlon)``. ``True`` cells are ignored.

        Returns
        -------
        numpy.ndarray
            One correlation value per time step.
        """
        assert self.obs.shape == self.sim.shape, "sim and obs must have identical dimensions"        

        ntime, nlat, nlon = self.sim.shape       

        # reshape
        sim_2d = self.sim.reshape(ntime, nlat * nlon)
        obs_2d = self.obs.reshape(ntime, nlat * nlon)        

        # mask
        if mask is not None:
            assert mask.shape == (nlat, nlon), "mask shape must match spatial dimensions"
            flat_mask = mask.flatten()
            sim_2d = sim_2d[:, ~flat_mask]
            obs_2d = obs_2d[:, ~flat_mask]
            
        pcc_array = np.full(ntime, np.nan)

        for t in range(ntime):
            sim_vec = sim_2d[t, :]
            obs_vec = obs_2d[t, :]

            valid_idx = (~np.isnan(sim_vec)) & (~np.isnan(obs_vec))

            if np.sum(valid_idx) > 1:
                pcc = np.corrcoef(sim_vec[valid_idx], obs_vec[valid_idx])[0, 1]
                pcc_array[t] = pcc
            else:
                pcc_array[t] = np.nan

        return pcc_array


class CategoricalEvaluationMetric:
    """Event-based verification metrics from a contingency table."""

    def __init__(self, sim, obs):
        """
        Initialize the metric class with simulated and observed values.

        Parameters
        ----------
        sim : array-like
            Simulated values.
        obs : array-like
            Observed values.
        """
        self.sim = np.array(sim)
        self.obs = np.array(obs)
        self.sim_bin = None
        self.obs_bin = None
        self.H = self.M = self.F = self.CN = None
    
    def binarize(self, threshold=0.0):
        """
        Convert continuous values to binary events based on threshold
        Values > threshold are considered 'event occurred' (1), else 'no event' (0).
        
        Parameters
        ----------
        threshold : float
            Threshold for event occurrence.
        """
        self.sim_bin = (self.sim > threshold).astype(int)
        self.obs_bin = (self.obs > threshold).astype(int)
        self._compute_contingency()

    def _compute_contingency(self):
        """Compute Hit, Miss, False Alarm, Correct Negative counts """
        self.H = np.sum((self.sim_bin == 1) & (self.obs_bin == 1))
        self.M = np.sum((self.sim_bin == 0) & (self.obs_bin == 1))
        self.F = np.sum((self.sim_bin == 1) & (self.obs_bin == 0))
        self.CN = np.sum((self.sim_bin == 0) & (self.obs_bin == 0))
        
    def POD(self):
        """ POD (Probability of Detection) """
        return self.H / (self.H + self.M) if (self.H + self.M) > 0 else None

    def FAR(self):
        """ FAR (False Alarm Ratio) """
        return self.F / (self.H + self.F) if (self.H + self.F) > 0 else None

    def CSI(self):
        """ CSI (Critical Success Index) """
        return self.H / (self.H + self.M + self.F) if (self.H + self.M + self.F) > 0 else None

    def HSS(self):
        """ HSS (Heidke Skill Score) """
        num = 2 * (self.H * self.CN - self.M * self.F)
        den = (self.H + self.M) * (self.M + self.CN) + (self.H + self.F) * (self.F + self.CN)
        return num / den if den > 0 else None

    def ETS(self):
        """ ETS (Equitable Threat Score) """
        H_random = (self.H + self.M) * (self.H + self.F) / (self.H + self.M + self.F + self.CN)
        den = self.H + self.M + self.F - H_random
        return (self.H - H_random) / den if den > 0 else None
        

class SignatureEvaluationMetric:
    """Hydrologic signature metrics based on discharge series."""

    def __init__(self, sim, obs):
        """
        Initialize the metric class with simulated and observed values.

        Parameters
        ----------
        sim : array-like
            Simulated values.
        obs : array-like
            Observed values.
        """
        self.sim = np.array(sim)
        self.obs = np.array(obs)
        
    def BiasRR(self, precip):
        """Compute runoff-ratio bias in percent.

        Parameters
        ----------
        precip : array-like
            Precipitation series aligned with ``sim`` and ``obs``.

        Returns
        -------
        float
            Relative runoff-ratio bias in percent.
        """
        rr_obs = self.obs.sum() / precip.sum()
        rr_sim = self.sim.sum() / precip.sum()
        return (rr_sim - rr_obs) / rr_obs * 100
        
    def BiasFHV(self, q_high=0.02):
        """
        High-flow volume bias (FHV), computed from the upper segment
        of the flow duration curve.
        """
        obs_sorted = np.sort(self.obs)[::-1]
        sim_sorted = np.sort(self.sim)[::-1]

        n = len(obs_sorted)
        p = np.arange(1, n + 1) / (n + 1)

        mask = p <= q_high

        return (
            sim_sorted[mask].sum() - obs_sorted[mask].sum()
        ) / obs_sorted[mask].sum() * 100
    
    def BiasFLV(self, q_low=0.7, eps=1e-6):
        """
        Low-flow volume bias (FLV), computed in log space.

        Parameters
        ----------
        q_low : float, optional (default=0.7)
            Quantile threshold defining low flows on the FDC
            (e.g., q_low=0.7 corresponds to the lowest 30% flows).
        eps : float, optional
            Small constant added to avoid log(0).

        Returns
        -------
        float
            Relative bias of low flows in log space.
        """
        # sort flows in descending order (FDC)
        obs_sorted = np.sort(self.obs)[::-1]
        sim_sorted = np.sort(self.sim)[::-1]
        
        n = len(obs_sorted)
        p = np.arange(1, n + 1) / (n + 1)

        mask = p >= q_low

        log_obs = np.log(obs_sorted[mask] + eps)
        log_sim = np.log(sim_sorted[mask] + eps)
        
        log_obs_min = log_obs.min()
        log_sim_min = log_sim.min()
        
        obs_area = (log_obs - log_obs_min).sum()
        sim_area = (log_sim - log_sim_min).sum()

        return (sim_area - obs_area) / obs_area * 100
    
    def BiasFMS(self, p1=0.2, p2=0.7, eps=1e-6):
        """
        Flow duration curve mid-segment slope bias (BiasFMS).

        Parameters
        ----------
        p1, p2 : float
            Exceedance probability bounds defining the mid-segment
            of the FDC (default: 0.2-0.7).
        eps : float
            Small constant to avoid log(0).

        Returns
        -------
        float
            Relative bias of the FDC mid-segment slope.
        """
        obs_sorted = np.sort(self.obs)[::-1]
        sim_sorted = np.sort(self.sim)[::-1]

        n = len(obs_sorted)
        p = np.arange(1, n + 1) / (n + 1)

        def _slope(q_sorted):
            q1 = np.interp(p1, p, q_sorted)
            q2 = np.interp(p2, p, q_sorted)
            return np.log(q1 + eps) - np.log(q2 + eps)

        s_obs = _slope(obs_sorted)
        s_sim = _slope(sim_sorted)

        return (s_sim - s_obs) / s_obs * 100
    
    def BiasFMM(self):
        """
        Flow duration curve median magnitude bias (BiasFMM).

        Defined as:
            (median(FDC_sim) - median(FDC_obs)) / median(FDC_obs)
        """
        fmm_obs = np.median(self.obs)
        fmm_sim = np.median(self.sim)

        return (fmm_sim - fmm_obs) / fmm_obs * 100
    
    @staticmethod
    def _compute_fdc(series):
        """
        Compute flow duration curve (FDC) for a given series.

        Parameters
        ----------
        series : array-like
            Streamflow series.

        Returns
        -------
        p : ndarray
            Exceedance probability (0-1).
        q : ndarray
            Sorted flow values (descending).
        """
        # remove NaN and non-positive values
        series = series[np.isfinite(series)]
        series = series[series > 0]

        if series.size == 0:
            raise ValueError("Input series contains no valid positive values.")

        # sort in descending order
        q = np.sort(series)[::-1]
        n = q.size

        # exceedance probability
        p = np.arange(1, n + 1) / (n + 1)

        return p, q

    def get_fdc(self, plot_bool=False):
        """
        Compute FDC curves for observed and simulated series.

        Returns
        -------
        fdc : dict
            Dictionary containing FDC data:
            {
                'obs': {'p': p_obs, 'q': q_obs},
                'sim': {'p': p_sim, 'q': q_sim}
            }
        
        example:
            plt.plot(r["obs"]["p"], r["obs"]["q"], "r-", label="Observed")
            plt.plot(r["sim"]["p"], r["sim"]["q"], "b-", label="sim")
            xs = [0.02, 0.2, 0.7]
            for x in xs:
                plt.axvline(x, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
                plt.text(
                    x+0.01,
                    plt.ylim()[1],
                    f"{x:.2f}",
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=9,
                    color="gray",
                    weight="bold",
                )
            plt.yscale("log")
            plt.legend()
            plt.xlim(0, 1)
            plt.ylabel("Discharge (m$^3$/s)")
            plt.xlabel("Flow exceedance probability [-]")
            plt.show(block=True)
        
        """
        p_obs, q_obs = self._compute_fdc(self.obs)
        p_sim, q_sim = self._compute_fdc(self.sim)

        if plot_bool:
            plt.plot(p_obs, q_obs, "r-", label="Observed")
            plt.plot(p_sim, q_sim, "b-", label="sim")
            xs = [0.02, 0.2, 0.7]
            for x in xs:
                plt.axvline(x, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
                plt.text(
                    x+0.01,
                    plt.ylim()[1],
                    f"{x:.2f}",
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=9,
                    color="gray",
                    weight="bold",
                )
            plt.yscale("log")
            plt.legend()
            plt.xlim(0, 1)
            plt.ylabel("Discharge (m$^3$/s)")
            plt.xlabel("Flow exceedance probability [-]")
            plt.show(block=True)
            
        return {
            "obs": {"p": p_obs, "q": q_obs},
            "sim": {"p": p_sim, "q": q_sim},
        }
        
    
def create_test_data(seed=42):
    """Create synthetic spatiotemporal fields for metric testing.

    Parameters
    ----------
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple of numpy.ndarray
        ``(reference, perturbed)`` with shape ``(T, H, W)``.
    """
    np.random.seed(seed)
    
    T, H, W = 10, 25, 25

    # Construct a base spatial pattern (e.g., high values concentrated in upper-right)
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian pattern centered at (0.8, 0.2)
    base_pattern = np.exp(-((X - 0.8)**2 + (Y - 0.2)**2) / 0.05)
    
    # Create time-dependent scaling (simulating seasonal or dynamic variation)
    time_scaling = np.sin(np.linspace(0, 2*np.pi, T)) + 1.5  # shifted sine wave to ensure all values >= 0

    # Construct data1: reference dataset with clean spatial-temporal structure
    data1 = np.array([time_scaling[t] * base_pattern for t in range(T)])  # shape: (T, H, W)

    # Construct data2: similar to data1 but with additional noise and a slight spatial bias
    noise_level = 0.2
    
    # Random spatial noise added to each time step
    spatial_noise = np.random.normal(0, noise_level, size=(T, H, W))
    
    # Small systematic bias in spatial structure (simulating model error)
    mode_shift = 0.03 * np.random.randn(H, W)

    # Combine to generate simulated dataset
    data2 = data1 + spatial_noise + mode_shift
    
    return data1, data2


def create_test_data2():
    """Create synthetic dataset pair with time-varying noise.

    Returns
    -------
    tuple of numpy.ndarray
        ``(obs_data, sim_data)`` with shape ``(n_time, n_lat, n_lon)``.
    """
    n_time = 30
    n_lat = 25
    n_lon = 25

    lat_grid, lon_grid = np.meshgrid(np.linspace(-1, 1, n_lat), np.linspace(-1, 1, n_lon))
    base_pattern = np.exp(-4 * (lat_grid**2 + lon_grid**2))

    obs_data = np.zeros((n_time, n_lat, n_lon))
    sim_data = np.zeros((n_time, n_lat, n_lon))

    for t in range(n_time):
        noise_level = 0.0
        if t < 5:
            noise_level = 0.05
        elif t < 25:
            noise_level = 0.3
        else:
            noise_level = 0.5

        obs_data[t] = base_pattern + np.random.normal(0, 0.02, size=base_pattern.shape)
        sim_data[t] = base_pattern + np.random.normal(0, noise_level, size=base_pattern.shape)
    
    return obs_data, sim_data


if __name__ == "__main__":
    data1, data2 = create_test_data2()
    
    EM = EvaluationMetric(data1, data2)
    
    ess = EM.ESS()
    

