# E91 Quantum Key Distribution Simulator
# Comprehensive simulation framework for E91 protocol over different channel types

# ========================== Import Libraries ==========================
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import math
from math import cos, sin, pi, sqrt, log2, exp,sinh
from enum import Enum
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class PhotonSource:
    def __init__(self, mu):
        self.mu = mu

    def photon_distribution(self, n_max=20):
        n_values = np.arange(n_max + 1)
        p_s = np.exp(-self.mu)*(self.mu**n_values)/np.array([math.factorial(n)for n in n_values])
        return p_s
    
# ========================== Channel Class ==========================
class ChannelType(Enum):
    FIBER = "fiber"
    FSO = "fso"  # Free Space Optical

from enum import Enum
from typing import Dict
from math import pi, exp
from scipy.constants import h, c  # Planck's constant and speed of light

class ChannelType(Enum):
    FIBER = "fiber"
    FSO = "fso"  # Free Space Optical

class QuantumChannel:
    """
    Quantum channel class supporting both fiber and FSO transmission
    """
    def __init__(self, channel_type: ChannelType, **kwargs):
        self.channel_type = channel_type
        self.user_params = kwargs  # Store user-defined parameters
        self._set_channel_parameters()

    def compute_dark_prob(self, dark_count_rate, time_window):
        """Compute probability of at least one dark count in a time window"""
        return 1 - exp(-dark_count_rate * time_window)
    
    def _set_channel_parameters(self):
        # Common dark count and window parameters (user-defined)
        dark_count_rate = self.user_params.get('dark_count_rate', 5000)  # Hz
        time_window = self.user_params.get('time_window', 1e-9)          # seconds
        p_dark = self.compute_dark_prob(dark_count_rate, time_window)


        """Set channel-specific parameters"""
        if self.channel_type == ChannelType.FIBER:
            self.alpha_db = self.user_params.get('alpha_db', 0.2)  # dB/km
            self.alpha_linear = self.alpha_db / 4.343               # 1/km
            self.max_distance = self.user_params.get('max_distance', 300)
            self.distance_km = self.user_params.get('distance_km', self.max_distance)

            # Raman parameters (user-defined or default)
            P_in = self.user_params.get('P_in', 1e-6)  # Input classical power [W]
            delta_lambda = self.user_params.get('delta_lambda', 0.7)  # nm
            rho = self.user_params.get('rho', 2e-9)   # Raman cross section [nm⁻¹·km⁻¹]
            eta_d = self.user_params.get('eta_d', 0.6)  # detector QE
            delta_t = self.user_params.get('delta_t', 1e-9)  # detection window [s]
            wavelength_nm = self.user_params.get('wavelength_nm', 1550)  # detection λ
            N_f = self.user_params.get('N_f', 2.0)  # scaling for forward scattering
            N_b = self.user_params.get('N_b', 2.0)  # scaling for backward scattering

            L = self.distance_km
            alpha = self.alpha_linear
            frequency = c / (wavelength_nm * 1e-9)
            E_photon = h * frequency

            # Outgoing classical power
            P_out = P_in * exp(-alpha * L)

            # Forward Raman power
            P_raman_f = N_f * P_out * L * rho * delta_lambda

            # Backward Raman power
            if alpha != 0:
                P_raman_b = N_b * P_out * (sinh(alpha * L) / alpha) * rho * delta_lambda
            else:
                P_raman_b = N_b * P_out * L * rho * delta_lambda  # fallback if lossless

            # Convert to photon count probabilities
            p_raman_f = (P_raman_f * delta_t * eta_d) / E_photon
            p_raman_b = (P_raman_b * delta_t * eta_d) / E_photon

            self.noise_components = {
                'raman_forward': p_raman_f,
                'raman_backward': p_raman_b,
                'dark': p_dark
            }

        
        elif self.channel_type == ChannelType.FSO:
            self.alpha_db = self.user_params.get('alpha_db', 0.1)   # dB/km
            self.max_distance = self.user_params.get('max_distance', 50)
            self.beam_divergence = self.user_params.get('beam_divergence', 0.025e-3)  # radians
            self.tx_aperture = self.user_params.get('tx_aperture', 0.01)  # meters
            self.rx_aperture = self.user_params.get('rx_aperture', 0.03)  # meters

            # Stray light parameters (user-defined or default)
            B_lambda = self.user_params.get('B_lambda', 1e-3)           # (sky radiance)W/nm/m²/sr
            delta_lambda = self.user_params.get('delta_lambda', 0.5)     # (Bandwidth)nm
            omega = self.user_params.get('omega', 1e-8)                 # Field of view (sr)
            eta_d = self.user_params.get('eta_d', 0.6)                  # quantum efficiency
            delta_t = self.user_params.get('delta_t', 1e-9)             # s
            wavelength_nm = self.user_params.get('wavelength_nm', 800) # nm
            eta_r = self.user_params.get('eta_d', 0.9)    

            stray_noise = self.compute_stray_noise(B_lambda, delta_lambda,
                                                   omega, eta_d, delta_t,
                                                   wavelength_nm,eta_r)

            self.noise_components = {
                'stray': stray_noise,
                'dark': p_dark
            }
        
        # Convert dB/km to 1/km for use in exp-based model
        self.alpha_linear = self.alpha_db / 4.343
        self.total_noise_prob = sum(self.noise_components.values())

    def compute_stray_noise(self, B_lambda: float, delta_lambda: float,
                            omega: float, eta_d: float, delta_t: float,
                            wavelength_nm: float,eta_r:float) -> float:
        """
        Compute stray photon probability per detection window.

        Returns:
            Stray photon count per window (dimensionless)
        """
        area = pi * (self.rx_aperture / 2) ** 2  # Aperture area in m²
        frequency = c / (wavelength_nm * 1e-9)   # Convert nm to Hz
        power = B_lambda * area * omega * delta_lambda * eta_r  # Power in W
        return (power *delta_t*eta_d ) / (h * frequency)  # Photon count

    def transmittance(self, distance_km: float) -> float:
        """
        Calculate channel transmittance for given distance
        """
        if self.channel_type == ChannelType.FIBER:
            return 10 ** (-self.alpha_db * distance_km / 10)
        
        elif self.channel_type == ChannelType.FSO:
            distance_m = distance_km * 1000
            geo_loss = (self.rx_aperture / (self.tx_aperture + self.beam_divergence * distance_m)) ** 2
            atm_loss = exp(-self.alpha_linear * distance_km)
            return geo_loss * atm_loss

    def get_channel_info(self) -> Dict:
        """Return channel configuration information"""
        return {
            'type': self.channel_type.value,
            'alpha_db': self.alpha_db,
            'max_distance': self.max_distance,
            'noise_components': self.noise_components,
            'total_noise': self.total_noise_prob
        }

    def update_parameters(self, **kwargs):
        """Update parameters and recompute noise if needed"""
        self.user_params.update(kwargs)
        self._set_channel_parameters()

# ========================== Detector Class ==========================
class QuantumDetector:
    """
    Quantum detector system for photon pair detection
    """
    
    def __init__(self, detector_efficiency: float = 0.6, collection_efficiency: float = 0.6):
        self.eta_detector = detector_efficiency      # Individual detector efficiency
        self.eta_collection = collection_efficiency  # Collection efficiency
        self.eta_total = self.eta_detector * self.eta_collection  # Total detection efficiency
        
        # Source parameters
        # self.pair_generation_rate = 0.64e6  # pairs per second
        
    def compute_detection_probabilities(self, transmittance: float) -> Dict[str, float]:
        """
        Compute detection probabilities for different photon number states
        
        Args:
            transmittance: Channel transmittance
            
        Returns:
            Dictionary with probabilities for different detection scenarios
        """
        T = transmittance
        
        # Photon number state probabilities after transmission
        p_both = T ** 2          # Both photons arrive
        p_single = 2 * T * (1 - T)  # One photon arrives
        p_none = (1 - T) ** 2    # No photons arrive
        
        return {
            'p_both_arrive': p_both,
            'p_single_arrive': p_single,
            'p_none_arrive': p_none
        }
    

        
    def compute_click_probability(self, noise_prob: float) -> float:
        """
        Compute probability of detector click (including noise)
        
        Args:
            noise_prob: Total noise probability per detector
            
        Returns:
            Click probability
        """
        base_click = self.eta_total + 2 * noise_prob * (1 - self.eta_total)
      
        return min(1.0, base_click)

    def compute_multi_photon_detection_probability(self, photons: int) -> float:
        if photons <= 0:
            return 0.0
        p_detect = 1 - (1 - self.eta_total) ** photons
        return min(1.0, p_detect)
    
    def compute_normalization_factor(self, transmittance: float, noise_prob: float, mu: float, distribution: Optional[np.ndarray] = None) -> float:
        """
        p_signalalization factor N using SPDC photon distribution and realistic detection.
        
        Args:
            transmittance: Combined channel transmittance (T)
            noise_prob: Total noise probability per detector
            mu: Mean photon number
            distribution: Optional photon number distribution to reuse
        
        Returns:
            Normalization factor N (0 ≤ N ≤ 1)
        """
        if distribution is None:
            source = PhotonSource(mu)
            distribution = source.photon_distribution(n_max=10)

        numerator_sum = 0
        denominator_sum = 0
        non_vaccum_prob = 1 - exp(-mu)*(1+mu)  # Approximate multi-pair error

        for n, p_n in enumerate(distribution):
            if n == 0:
                continue

            p_signal = self.compute_multi_photon_detection_probability(n)
            signal_click_prob = p_signal
            total_click_prob = signal_click_prob + 2 * noise_prob * (1 - signal_click_prob)

            numerator_sum =  (transmittance ** 2) * (signal_click_prob ** 2) * (1 - non_vaccum_prob)

            denominator_sum =  (
                (transmittance ** 2) * (total_click_prob ** 2) +
                4 * transmittance * (1 - transmittance) * noise_prob * total_click_prob +
                4 * ((1 - transmittance) ** 2) * (noise_prob ** 2)
            )

        return numerator_sum / denominator_sum if denominator_sum > 0 else 0.0

    
    def update_efficiency(self, detector_eff: Optional[float] = None, 
                         collection_eff: Optional[float] = None):
        """Update detector efficiencies"""
        if detector_eff is not None:
            self.eta_detector = detector_eff
        if collection_eff is not None:
            self.eta_collection = collection_eff
        self.eta_total = self.eta_detector * self.eta_collection    

# ========================== E91 Simulator Class ==========================
class E91Simulator:
    """
    Complete E91 quantum key distribution simulator
    Supports center-source configuration with physics realism.
    """

    def __init__(self, channel: QuantumChannel, detector: QuantumDetector, distance_km: float, mu: float = 0.1, f_rep: float = 1e6):
        self.channel = channel
        self.detector = detector
        self.distance_km = distance_km  # store for reuse
        self.f_rep = f_rep
        self.mu = mu  # Source parameter for photon distribution
        self.spdc_source = PhotonSource(mu=self.mu)  # SPDC source with mean photon number mu
        self.detector.pair_generation_rate = self.mu * self.f_rep


        # Bell test measurement settings (CHSH inequality)
        self.measurement_angles = {
            'alice_1': 0,
            'alice_2': pi / 4,
            'bob_1': -pi / 8,
            'bob_2': pi / 8
        }

        self.entanglement_phase = pi
        self.simulation_results = {}

        #  Add center-source support:
        half_distance = distance_km / 2

        self.channel_alice = QuantumChannel(channel.channel_type)
        self.channel_bob = QuantumChannel(channel.channel_type)

        # Copy configuration and update distances
        self.channel_alice.update_parameters(**channel.get_channel_info())
        self.channel_bob.update_parameters(**channel.get_channel_info())


    
    @staticmethod
    def binary_entropy(x: float) -> float:
        """
        Calculate binary entropy function H(x) = -x*log₂(x) - (1-x)*log₂(1-x)
        
        Args:
            x: Probability value between 0 and 1
            
        Returns:
            Binary entropy value
        """
        epsilon = 1e-12
        x = np.clip(x, epsilon, 1 - epsilon)
        return -x * log2(x) - (1 - x) * log2(1 - x)
    
    def correlation_function(self, theta_alice: float, theta_bob: float, 
                           normalization: float, phase: float) -> float:
        """
        Calculate correlation function E(θₐ, θᵦ) for given measurement angles
        
        Args:
            theta_alice: Alice's measurement angle
            theta_bob: Bob's measurement angle
            normalization: Normalization factor N
            phase: Entanglement phase
            
        Returns:
            Correlation value
        """
        term1 = -cos(2 * theta_alice) * cos(2 * theta_bob)
        term2 = cos(phase) * sin(2 * theta_alice) * sin(2 * theta_bob)
        return normalization * (term1 + term2)
    
    def compute_bell_parameter(self, normalization: float, phase: float) -> float:
        """
        Compute Bell parameter S for CHSH inequality
        
        Args:
            normalization: Normalization factor N
            phase: Entanglement phase
            
        Returns:
            Bell parameter S
        """
        angles = self.measurement_angles
        
        # Calculate four correlation functions
        E11 = self.correlation_function(angles['alice_1'], angles['bob_1'], 
                                       normalization, phase)
        E12 = self.correlation_function(angles['alice_1'], angles['bob_2'], 
                                       normalization, phase)
        E21 = self.correlation_function(angles['alice_2'], angles['bob_1'], 
                                       normalization, phase)
        E22 = self.correlation_function(angles['alice_2'], angles['bob_2'], 
                                       normalization, phase)
        
        # CHSH combination: S = |E₁₁ + E₁₂ - E₂₁ + E₂₂|
        S = abs(E11 + E12 - E21 + E22)
        
        return S
    
    def compute_qber(self, bell_parameter: float) -> float:
        """
        Compute Quantum Bit Error Rate (QBER) from Bell parameter
        
        Args:
            bell_parameter: Bell parameter S
            
        Returns:
            QBER value
        """
        # Clip S to valid range [0, 2√2]
        S = np.clip(bell_parameter, 0, 2 * sqrt(2))
        return 0.5 * (1 - S / (2 * sqrt(2)))
    
    def compute_secret_key_rate(self, bell_parameter: float, qber: float, 
                              transmittance: float) -> float:
        """
        Compute Secret Key Rate using Acín et al.'s formula
        
        Args:
            bell_parameter: Bell parameter S
            qber: Quantum bit error rate
            transmittance: Channel transmittance
            
        Returns:
            Secret key rate in bits per second
        """
        S = bell_parameter
        
        # No key extraction possible if S ≤ 2 (no Bell violation)
        if S <= 2:
            return 0
        
        # Acín et al. formula term
        term = (1 + sqrt(S**2 / 4 - 1)) / 2
        
        # SKR = (1/3) * ν * T * [1 - H(Q) - H(term)]
        skr = (1/3) * self.mu*self.f_rep * transmittance * \
              (1 - self.binary_entropy(qber) - self.binary_entropy(term))
        
        return max(0, skr)  # Ensure non-negative
    
    def simulate_single_distance(self, distance_km: float) -> Dict:
        """
        Simulate E91 protocol for a single distance with optional eavesdropping.

        Args:
            distance_km: Distance in kilometers
       
        
        Returns:
            Dictionary of simulation results
        """
        # Channel transmittance for center-source setup
        T_A = self.channel_alice.transmittance(distance_km / 2)
        T_B = self.channel_bob.transmittance(distance_km / 2)
        T = (T_A * T_B)

        # SPDC photon number distribution (mean μ already set in self.spdc_source)
        distribution = self.spdc_source.photon_distribution(n_max=10)

        p_noise = self.channel.total_noise_prob

        N = self.detector.compute_normalization_factor(
        transmittance=T,
        noise_prob=p_noise,
        mu=self.mu,
        distribution=distribution
        )

     

        # Bell parameter, QBER, SKR
        S = self.compute_bell_parameter(N, self.entanglement_phase)
        Q = self.compute_qber(S)
        Q = min(0.5, Q)

        SKR = self.compute_secret_key_rate(S, Q, T)

        return {
            'distance': distance_km,
            'transmittance': T,
            'normalization': N,
            'bell_parameter': S,
            'qber': Q,
            'qber_percent': Q * 100,
            'secret_key_rate': SKR,
            'bell_violation': S > 2,
            'secure_communication': S > 2 and Q < 0.146
        }
        
    def simulate_distance_range(self, min_distance: float = 0.01, 
                              max_distance: Optional[float] = None,
                              num_points: int = 300) -> Dict:
        """
        Simulate E91 protocol over a range of distances
        
        Args:
            min_distance: Minimum distance in km
            max_distance: Maximum distance in km (uses channel max if None)
            num_points: Number of distance points to simulate
            
        Returns:
            Dictionary with arrays of simulation results
        """
        if max_distance is None:
            max_distance = self.channel.max_distance
        
        distances = np.linspace(min_distance, max_distance, num_points)
        
        results = {
            'distances': distances,
            'transmittances': [],
            'bell_parameters': [],
            'qbers': [],
            'qber_percents': [],
            'secret_key_rates': [],
            'bell_violations': [],
            'secure_regions': []
        }
        
        for distance in distances:
            single_result = self.simulate_single_distance(distance)
            
            results['transmittances'].append(single_result['transmittance'])
            results['bell_parameters'].append(single_result['bell_parameter'])
            results['qbers'].append(single_result['qber'])
            results['qber_percents'].append(single_result['qber_percent'])
            results['secret_key_rates'].append(single_result['secret_key_rate'])
            results['bell_violations'].append(single_result['bell_violation'])
            results['secure_regions'].append(single_result['secure_communication'])
        
        # Convert lists to numpy arrays for easier manipulation
        for key in results:
            if key != 'distances':
                results[key] = np.array(results[key])
        
        # Store results for later use
        self.simulation_results = results
        
        return results
    
    def update_measurement_angles(self, **angles):
        """
        Update measurement angles for Bell test
        
        Args:
            **angles: Keyword arguments for angle updates
                     (alice_1, alice_2, bob_1, bob_2)
        """
        for key, value in angles.items():
            if key in self.measurement_angles:
                self.measurement_angles[key] = value
    
    def get_simulation_summary(self) -> Dict:
        """Get summary of current simulation setup"""
        return {
            'channel_info': self.channel.get_channel_info(),
            'detector_efficiency': self.detector.eta_total,
            'measurement_angles': self.measurement_angles,
            'entanglement_phase': self.entanglement_phase
        }

# ========================== Example Usage ==========================
print("E91 QKD Simulator Classes Initialized Successfully!")
print("\nAvailable Classes:")
print("- QuantumChannel: Models fiber and FSO channels")
print("- QuantumDetector: Simulates photon detection systems") 
print("- E91Simulator: Complete E91 protocol simulation")
print("\nExample instantiation:")
print("channel = QuantumChannel(ChannelType.FIBER)")
print("detector = QuantumDetector()")
print("simulator = E91Simulator(channel, detector)")


# ========================== Plotting Functions ==========================
def plot_qber_vs_distance(distance_range=(1, 150), num_points=50,
                          detector_efficiency=0.6, collection_efficiency=0.6,
                          channel_type="fiber", 
                          receiver_diameter=0.03, transmitter_diameter=0.01,
                          beam_divergence=0.025e-3, atmospheric_attenuation=0.1,
                          mu=0.1, f_rep=1e6):  # 

    distances = np.linspace(*distance_range, num_points)
    qbers = []

    for d in distances:
        channel = QuantumChannel(ChannelType.FIBER if channel_type == "fiber" else ChannelType.FSO)
        if channel_type == "fso":
            channel.update_parameters(
                rx_aperture=receiver_diameter,
                tx_aperture=transmitter_diameter,
                beam_divergence=beam_divergence,
                alpha_db=atmospheric_attenuation
            )

        detector = QuantumDetector(detector_efficiency, collection_efficiency)

        sim = E91Simulator(channel, detector, distance_km=d, mu=mu, f_rep=f_rep)  # 
        result = sim.simulate_single_distance(d)
        qbers.append(result['qber_percent'])

    plt.figure(figsize=(10, 6))
    plt.plot(distances, qbers, linewidth=3.5, markersize=6, label='QBER')
    plt.axhline(14.6, color='red',  linewidth=3.5, linestyle='--', label='14.6% Threshold')
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=18)
    plt.ylabel('QBER (%)', fontsize=18)
    plt.title(f'QBER vs Distance ({channel_type.upper()})', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return distances, qbers

def plot_skr_vs_distance(distance_range=(1, 150), num_points=50,
                         detector_efficiency=0.6, collection_efficiency=0.6,
                         channel_type="fiber", 
                         receiver_diameter=0.03, transmitter_diameter=0.01,
                         beam_divergence=0.025e-3, atmospheric_attenuation=0.1,
                         mu=0.1, f_rep=1e6):
    """
    Plot Secret Key Rate (SKR) vs distance for E91 protocol.
    """
    distances = np.linspace(*distance_range, num_points)
    skr_values = []

    for d in distances:
        channel = QuantumChannel(ChannelType.FIBER if channel_type == "fiber" else ChannelType.FSO)
        if channel_type == "fso":
            channel.update_parameters(
                rx_aperture=receiver_diameter,
                tx_aperture=transmitter_diameter,
                beam_divergence=beam_divergence,
                alpha_db=atmospheric_attenuation
            )

        detector = QuantumDetector(detector_efficiency, collection_efficiency)

        sim = E91Simulator(channel, detector, distance_km=d, mu=mu, f_rep=f_rep)  # 
        result = sim.simulate_single_distance(d)
        skr_values.append(result['secret_key_rate'])

    plt.figure(figsize=(10, 6))
    plt.plot(distances, skr_values, linewidth=3.5, markersize=6, label='SKR')
    plt.semilogy()
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=18)
    plt.ylabel('Secret Key Rate (bps)', fontsize=18)
    plt.title(f'SKR vs Distance ({channel_type.upper()})', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return distances, skr_values

def plot_bell_violation_vs_distance(distance_range=(1, 150), num_points=50,
                                    detector_efficiency=0.6, collection_efficiency=0.6,
                                    channel_type="fiber", 
                                    receiver_diameter=0.03, transmitter_diameter=0.01,
                                    beam_divergence=0.025e-3, atmospheric_attenuation=0.1,
                                    mu=0.1, f_rep=1e6):
    """
    Plot Bell parameter S vs distance for E91 protocol.
    """
    distances = np.linspace(*distance_range, num_points)
    S_values = []

    for d in distances:
        channel = QuantumChannel(ChannelType.FIBER if channel_type == "fiber" else ChannelType.FSO)
        if channel_type == "fso":
            channel.update_parameters(
                rx_aperture=receiver_diameter,
                tx_aperture=transmitter_diameter,
                beam_divergence=beam_divergence,
                alpha_db=atmospheric_attenuation
            )

        detector = QuantumDetector(detector_efficiency, collection_efficiency)

        sim = E91Simulator(channel, detector, distance_km=d, mu=mu, f_rep=f_rep)
        result = sim.simulate_single_distance(d)
        S_values.append(result['bell_parameter'])

    plt.figure(figsize=(10, 6))
    plt.plot(distances, S_values, linewidth=3.5, markersize=6, label='S')
    plt.axhline(2, color='red',linewidth=3.5, linestyle='--', label='Classical Limit (S=2)')
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=18)
    plt.ylabel('Bell Parameter S', fontsize=18)
    plt.title(f'Bell Violation S vs Distance ({channel_type.upper()})', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return distances, S_values

def plot_qber_vs_mu(mu_range=(0.01, 1.0), num_points=50,
                    fixed_distance_km=120,
                    f_rep=1e6,
                    detector_efficiency=0.6, collection_efficiency=0.6,
                    channel_type="fiber"):
    """
    Plot QBER vs mean photon number μ for E91 protocol.
    Runs standalone if called without parameters.
    """
    mu_values = np.linspace(*mu_range, num_points)
    qber_values = []

    for mu in mu_values:
        channel = QuantumChannel(ChannelType.FIBER if channel_type == "fiber" else ChannelType.FSO)
        detector = QuantumDetector(detector_efficiency, collection_efficiency)

        simulator = E91Simulator(channel, detector, distance_km=fixed_distance_km, mu=mu, f_rep=f_rep)
        result = simulator.simulate_single_distance(fixed_distance_km)

        qber_values.append(result['qber_percent'])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, linewidth=3.5, markersize=5)
    plt.axhline(14.6, color='red',linewidth=3.5, linestyle='--', label='QBER Threshold (14.6%)')
    plt.grid(True)
    plt.xlabel('Mean Photon Number μ', fontsize=18)
    plt.ylabel('QBER (%)', fontsize=18)
    plt.title(f'QBER vs μ at {fixed_distance_km} km ({channel_type.upper()})', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

    return mu_values, qber_values

def plot_skr_vs_mu(mu_range=(0.01, 1.0), num_points=50,
                   fixed_distance_km=120,
                   f_rep=1e6,
                   detector_efficiency=0.6, collection_efficiency=0.6,
                   channel_type="fiber"):
    """
    Plot SKR vs mean photon number μ for E91 protocol.
    """
    mu_values = np.linspace(*mu_range, num_points)
    skr_values = []

    for mu in mu_values:
        channel = QuantumChannel(ChannelType.FIBER if channel_type == "fiber" else ChannelType.FSO)
        detector = QuantumDetector(detector_efficiency, collection_efficiency)

        simulator = E91Simulator(channel, detector, distance_km=fixed_distance_km, mu=mu, f_rep=f_rep)
        result = simulator.simulate_single_distance(fixed_distance_km)

        skr_values.append(result['secret_key_rate'])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, linewidth=3.5, markersize=5)
    plt.grid(True)
    plt.xlabel('Mean Photon Number μ', fontsize=18)
    plt.ylabel('Secret Key Rate (bps)', fontsize=18)
    plt.title(f'SKR vs μ at {fixed_distance_km} km ({channel_type.upper()})', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return mu_values, skr_values

def plot_bell_violation_vs_mu(mu_range=(0.01, 1.0), num_points=50,
                              fixed_distance_km=120,
                              f_rep=1e6,
                              detector_efficiency=0.6, collection_efficiency=0.6,
                              channel_type="fiber"):
    """
    Plot Bell parameter S vs mean photon number μ for E91 protocol.
    """
    mu_values = np.linspace(*mu_range, num_points)
    S_values = []

    for mu in mu_values:
        channel = QuantumChannel(ChannelType.FIBER if channel_type == "fiber" else ChannelType.FSO)
        detector = QuantumDetector(detector_efficiency, collection_efficiency)

        simulator = E91Simulator(channel, detector, distance_km=fixed_distance_km, mu=mu, f_rep=f_rep)
        result = simulator.simulate_single_distance(fixed_distance_km)

        S_values.append(result['bell_parameter'])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, S_values, linewidth=3.5, markersize=5, label='Bell Parameter S')
    plt.axhline(2, color='red',  linewidth=3.5, linestyle='--', label='Classical Limit (S=2)')
    plt.grid(True)
    plt.xlabel('Mean Photon Number μ', fontsize=18)
    plt.ylabel('Bell Parameter S', fontsize=18)
    plt.title(f'Bell Violation S vs μ at {fixed_distance_km} km ({channel_type.upper()})', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return mu_values, S_values


def compare_fiber_fso_channels(mu=0.1, f_rep=1e6, 
                                fiber_distance=200, fso_distance=50, 
                                num_points=300, verbose=True):
    """
    Simulates E91 protocol over both Fiber and FSO quantum channels and compares:
    - QBER
    - Bell Parameter
    - Secret Key Rate
    - Transmittance

    Args:
        mu (float): Mean photon number per pulse.
        f_rep (float): Repetition rate (Hz).
        fiber_distance (float): Max distance for fiber simulation (km).
        fso_distance (float): Max distance for FSO simulation (km).
        num_points (int): Number of simulation points across distance range.
        verbose (bool): Whether to print summary results.

    Returns:
        Tuple of results: (fiber_results, fso_results)
    """
    # Initialize components
    fiber_channel = QuantumChannel(ChannelType.FIBER)
    fso_channel = QuantumChannel(ChannelType.FSO)
    detector = QuantumDetector()

    # Initialize simulators
    fiber_sim = E91Simulator(fiber_channel, detector, distance_km=fiber_distance, mu=mu, f_rep=f_rep)
    fso_sim = E91Simulator(fso_channel, detector, distance_km=fso_distance, mu=mu, f_rep=f_rep)

    # Run simulations
    fiber_results = fiber_sim.simulate_distance_range(min_distance=0.1, max_distance=fiber_distance, num_points=num_points)
    fso_results = fso_sim.simulate_distance_range(min_distance=0.1, max_distance=fso_distance, num_points=num_points)

    # Plot comparisons
    plt.figure(figsize=(15, 10))

    # QBER vs Distance
    plt.subplot(2, 2, 1)
    plt.plot(fiber_results['distances'], fiber_results['qber_percents'], label='Fiber', color='blue')
    plt.plot(fso_results['distances'], fso_results['qber_percents'], label='FSO', color='green')
    plt.axhline(14.6, color='red', linestyle='--', alpha=0.7, label='Threshold (14.6%)')
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title('QBER vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 50)

    # Bell Parameter
    plt.subplot(2, 2, 2)
    plt.plot(fiber_results['distances'], fiber_results['bell_parameters'], label='Fiber', color='blue')
    plt.plot(fso_results['distances'], fso_results['bell_parameters'], label='FSO', color='green')
    plt.axhline(2, color='red', linestyle='--', alpha=0.7, label='Classical Limit')
    plt.axhline(2*sqrt(2), color='orange', linestyle='--', alpha=0.7, label='Quantum Limit')
    plt.xlabel('Distance (km)')
    plt.ylabel('Bell Parameter S')
    plt.title('Bell Parameter vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SKR
    plt.subplot(2, 2, 3)
    plt.semilogy(fiber_results['distances'], fiber_results['secret_key_rates'], label='Fiber', color='blue')
    plt.semilogy(fso_results['distances'], fso_results['secret_key_rates'], label='FSO', color='green')
    plt.xlabel('Distance (km)')
    plt.ylabel('SKR (bits/s)')
    plt.title('Secret Key Rate vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Transmittance
    plt.subplot(2, 2, 4)
    plt.semilogy(fiber_results['distances'], fiber_results['transmittances'], label='Fiber', color='blue')
    plt.semilogy(fso_results['distances'], fso_results['transmittances'], label='FSO', color='green')
    plt.xlabel('Distance (km)')
    plt.ylabel('Transmittance')
    plt.title('Transmittance vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    if verbose:
        print("=== E91 QKD Simulation Results ===")
        print(f"\nFiber (up to {fiber_distance} km):")
        print(f"  QBER at max: {fiber_results['qber_percents'][-1]:.2f}%")
        print(f"  Bell Param at max: {fiber_results['bell_parameters'][-1]:.3f}")
        if len(fiber_results['secure_regions']) > 0:
            max_sec = fiber_results['distances'][fiber_results['secure_regions']][-1]
            print(f"  Secure range: up to {max_sec:.1f} km")
        else:
            print("  No secure communication possible.")

        print(f"\nFSO (up to {fso_distance} km):")
        print(f"  QBER at max: {fso_results['qber_percents'][-1]:.2f}%")
        print(f"  Bell Param at max: {fso_results['bell_parameters'][-1]:.3f}")
        if len(fso_results['secure_regions']) > 0:
            max_sec = fso_results['distances'][fso_results['secure_regions']][-1]
            print(f"  Secure range: up to {max_sec:.1f} km")
        else:
            print("  No secure communication possible.")

    return fiber_results, fso_results