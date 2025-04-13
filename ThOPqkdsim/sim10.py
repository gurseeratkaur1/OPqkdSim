import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import math

class DPSQKDSimulator:
    """
    Simulator for Differential Phase Shift (DPS) Quantum Key Distribution protocol.
    """
    def __init__(self, 
                 nem=10**6,              # Number of blocks sent by Alice
                 repetition_rate=1e9,    # Pulse repetition rate (Hz)
                 mu=0.1,                 # Mean photon number per pulse
                 ebit=0.01,              # Bit error rate
                 eta_det=0.1,            # Detector efficiency
                 dark_count=1e-7,        # Dark count probability per detection window
                 alpha=0.2,              # Fiber loss coefficient (dB/km)
                 distance=50,            # Channel distance (km)
                 delta_bs1=0.005,        # BS1 transmittance deviation from ideal 50%
                 delta_bs2=0.005,        # BS2 transmittance deviation from ideal 50%
                 t_prob=0.5):            # Probability for code event selection
        """
        Initialize the DPS QKD simulator with given parameters.
        """
        self.nem = nem                   # Number of total blocks sent by Alice
        self.repetition_rate = repetition_rate  # Pulse repetition rate in Hz
        self.mu = mu                     # Mean photon number per pulse
        self.ebit = ebit                 # Bit error rate
        self.eta_det = eta_det           # Detector efficiency
        self.dark_count = dark_count     # Dark count probability
        self.alpha = alpha               # Fiber loss coefficient (dB/km)
        self.distance = distance         # Channel distance (km)
        self.delta_bs1 = delta_bs1       # BS1 transmittance deviation
        self.delta_bs2 = delta_bs2       # BS2 transmittance deviation
        self.t_prob = t_prob             # Probability for code event selection
        
        # Calculate derived parameters
        self.eta_ch = 10**(-self.alpha * self.distance / 10)  # Channel transmittance
        self.eta = self.eta_ch * self.eta_det                 # Overall transmittance
        
        # BS transmittance ranges
        self.eta1_L = 0.5 - self.delta_bs1
        self.eta1_U = 0.5 + self.delta_bs1
        self.eta2_L = 0.5 - self.delta_bs2
        self.eta2_U = 0.5 + self.delta_bs2
        
        # Use upper bounds for security analysis
        self.eta1 = self.eta1_U
        self.eta2 = self.eta2_U
        
        # Calculate lambda(eta1_U, eta2_U) according to equation (16)
        self.lambda_factor = self._calculate_lambda(self.eta1, self.eta2)
    
    def _calculate_lambda(self, eta1, eta2):
        """
        Calculate the lambda factor according to equation (16) in the paper.
        """
        term1 = 1 - (1 - eta1) * eta2
        term2 = np.sqrt((term1)**2 - 4 * eta1 * (1 - eta1) * (1 - eta2)**2)
        numerator = term1 + term2
        denominator = 2 * (1 - eta1) * (1 - eta2)**2
        return numerator / denominator
    
    def _calculate_qn(self, n):
        """
        Calculate qn, the probability of emitting n or more photons per block.
        """
        # For weak coherent pulses, qn is Poisson distributed
        # Each block has 3 pulses with mean photon number mu per pulse
        total_mu = 3 * self.mu
        # Probability of having n or more photons in the block
        qn = 1 - np.sum([np.exp(-total_mu) * (total_mu)**m / math.factorial(m) for m in range(n)])
        return qn
    
    def _calculate_detection_rate(self, mu=None):
        """
        Calculate the detection rate Q according to the paper.
        """
        if mu is None:
            mu = self.mu
            
        # Detection probability per block (simplified model based on Poisson statistics)
        # Q = 2ηµe^(-2ηµ) as specified in the paper
        Q = 2 * self.eta * mu * np.exp(-2 * self.eta * mu)
        return Q
    
    def _calculate_upper_bound_phase_error(self, Q, mu=None):
        """
        Calculate upper bound on phase error rate based on Theorem 3.
        """
        if mu is None:
            mu = self.mu
            
        # Save current mu value
        original_mu = self.mu
        
        # Temporarily set mu to calculate qn
        self.mu = mu
        
        # Calculate probabilities for different photon number emissions
        q1 = self._calculate_qn(1)
        q2 = self._calculate_qn(2)
        q3 = self._calculate_qn(3)
        
        # Restore original mu
        self.mu = original_mu
        
        # Calculate upper bound on phase error rate according to equation (15)
        e_U_ph = (self.lambda_factor * (self.ebit + np.sqrt(q1 * q3) / Q + 2 * self.delta_bs2) 
                 + q2 / Q)
        return e_U_ph
    
    def calculate_secret_key_rate(self, mu=None):
        """
        Calculate the secret key rate per pulse.
        """
        if mu is None:
            mu = self.mu
        
        # Detection rate
        Q = self._calculate_detection_rate(mu)
        
        # Upper bound on phase error rate
        e_U_ph = self._calculate_upper_bound_phase_error(Q, mu)
        
        # Number of code events
        Ncode = self.t_prob * self.nem * Q
        
        # Cost of error correction - based on Shannon limit
        NEC = Ncode * self._shannon_entropy(self.ebit)
        
        # Amount of privacy amplification
        NPA = Ncode * self._shannon_entropy(e_U_ph)
        
        # Secret key length
        ell = Ncode - NPA - NEC
        
        # Secret key rate per emitted pulse (equation 17)
        R = ell / (3 * self.nem)
        
        # Secret key rate in bits per second
        R_bps = R * self.repetition_rate
        
        return R_bps if R > 0 else 0
    
    def _shannon_entropy(self, p):
        """
        Shannon entropy function h(p) = -p*log2(p) - (1-p)*log2(1-p)
        """
        if p <= 0 or p >= 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    def calculate_qber(self):
        """
        Calculate the expected QBER based on dark counts and signal.
        """
        # Calculate signal and noise contributions
        # Signal detection probability
        signal_prob = self.eta * self.mu * np.exp(-self.eta * self.mu)
        
        # Noise due to dark counts
        noise_prob = self.dark_count
        
        # QBER calculation: probability of error events / probability of detection events
        # Errors occur due to dark counts and imperfect interference
        if signal_prob + noise_prob > 0:
            qber = (0.5 * noise_prob + self.ebit * signal_prob) / (signal_prob + noise_prob)
        else:
            qber = 0.5  # No signal, only random noise
            
        return qber
    
    
    def get_qber_vs_mu_data(self, mu_range=None, points=100):
        """
        Get data for QBER vs mean photon number.
        """
        if mu_range is None:
            mu_range = (0.001, 1.0)
        
        mu_values = np.linspace(mu_range[0], mu_range[1], points)
        qber_values = []
        
        original_mu = self.mu
        
        for mu in mu_values:
            self.mu = mu
            qber_values.append(self.calculate_qber())
        
        self.mu = original_mu  # Restore original value
        
        return mu_values, qber_values
    
    def get_qber_vs_distance_data(self, distance_range=None, points=100):
        """
        Get data for QBER vs distance.
        """
        if distance_range is None:
            distance_range = (0, 200)
        
        distance_values = np.linspace(distance_range[0], distance_range[1], points)
        qber_values = []
        
        original_distance = self.distance
        original_eta_ch = self.eta_ch
        original_eta = self.eta
        
        for distance in distance_values:
            self.distance = distance
            self.eta_ch = 10**(-self.alpha * self.distance / 10)
            self.eta = self.eta_ch * self.eta_det
            qber_values.append(self.calculate_qber())
        
        # Restore original values
        self.distance = original_distance
        self.eta_ch = original_eta_ch
        self.eta = original_eta
        
        return distance_values, qber_values

    def get_skr_vs_mu_data(self, mu_range=None, points=100):
        """
        Get data for Secret Key Rate vs mean photon number.
        """
        if mu_range is None:
            mu_range = (0.001, 1.0)
        
        mu_values = np.linspace(mu_range[0], mu_range[1], points)
        skr_values = []
        
        for mu in mu_values:
            # Calculate SKR directly without modifying self.mu
            skr = self.calculate_secret_key_rate(mu)
            skr_values.append(skr)
        
        return mu_values, skr_values

    def get_skr_vs_distance_data(self, distance_range=None, points=100):
        """
        Get data for Secret Key Rate vs distance.
        """
        if distance_range is None:
            distance_range = (0, 200)
        
        distance_values = np.linspace(distance_range[0], distance_range[1], points)
        skr_values = []
        optimal_mu_values = []
        
        original_distance = self.distance
        original_mu = self.mu
        original_eta_ch = self.eta_ch
        original_eta = self.eta
        
        for distance in distance_values:
            self.distance = distance
            self.eta_ch = 10**(-self.alpha * self.distance / 10)
            self.eta = self.eta_ch * self.eta_det
            skr_values.append(self.calculate_secret_key_rate())
        
        # Restore original values
        self.distance = original_distance
        self.mu = original_mu
        self.eta_ch = original_eta_ch
        self.eta = original_eta

        return distance_values, skr_values

    def plot_qber_vs_mu(self, mu_range=None, points=100):
        """
        Plot QBER vs mean photon number.
        """
        mu_values, qber_values = self.get_qber_vs_mu_data(mu_range, points)
        
        plt.figure(figsize=(10, 6))
        plt.plot(mu_values, qber_values, 'b-', linewidth=2)
        plt.xlabel('Mean Photon Number (μ)', fontsize=12)
        plt.ylabel('QBER', fontsize=12)
        plt.title('QBER vs Mean Photon Number', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        
    def plot_qber_vs_distance(self, distance_range=None, points=100):
        """
        Plot QBER vs distance.
        """
        distance_values, qber_values = self.get_qber_vs_distance_data(distance_range, points)
        
        plt.figure(figsize=(10, 6))
        plt.plot(distance_values, qber_values, 'g-', linewidth=2)
        plt.xlabel('Distance (km)', fontsize=12)
        plt.ylabel('QBER', fontsize=12)
        plt.title('QBER vs Distance', fontsize=14)
        plt.grid(True)
        plt.tight_layout()

    def plot_skr_vs_mu(self, mu_range=None, points=100):
        """
        Plot Secret Key Rate vs mean photon number - should show bell curve shape.
        """
        if mu_range is None:
            # Use narrower range to better visualize the bell curve
            mu_range = (0.001, 0.5)
        
        mu_values, skr_values = self.get_skr_vs_mu_data(mu_range, points)
        
        plt.figure(figsize=(10, 6))
        plt.plot(mu_values, skr_values, 'r-', linewidth=2)
        plt.xlabel('Mean Photon Number (μ)', fontsize=12)
        plt.ylabel('SKR (bits/s)', fontsize=12)
        plt.title('Secret Key Rate vs Mean Photon Number', fontsize=14)
        plt.grid(True)
        plt.tight_layout()

    def plot_skr_vs_distance(self, distance_range=None, points=100, optimize_mu=True):
        """
        Plot Secret Key Rate vs distance.
        """
        if optimize_mu:
            distance_values, skr_values = self.get_skr_vs_distance_data(distance_range, points)
        else:
            distance_values, skr_values = self.get_skr_vs_distance_data(distance_range, points)
        
        plt.figure(figsize=(10, 6))
        plt.plot(distance_values, skr_values, 'c-', linewidth=2)
        plt.xlabel('Distance (km)', fontsize=12)
        plt.ylabel('SKR (bits/s)', fontsize=12)
        plt.title('Secret Key Rate vs Distance', fontsize=14)
        plt.grid(True)
        plt.tight_layout()

    def print_summary(self):
        """
        Print a summary of the simulation results.
        """
        # Print summary of results
        print("DPS QKD Simulation Summary:")
        print(f"Distance: {self.distance} km")
        print(f"Mean photon number (μ): {self.mu}")
        print(f"Channel transmittance: {self.eta_ch:.6f}")
        print(f"Overall transmittance: {self.eta:.6f}")
        print(f"Detection rate (Q): {self._calculate_detection_rate():.6f}")
        print(f"QBER: {self.calculate_qber():.6f}")
        print(f"Secret Key Rate: {self.calculate_secret_key_rate():.2f} bits/s")