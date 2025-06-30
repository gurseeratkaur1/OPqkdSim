import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


class PhotonSource:
    """
    Simulates a coherent light source for COW protocol with controllable mean photon number.
    """
    def __init__(self, mu):
        """
        Initialize the photon source with mean photon number mu.
        
        Args:
            mu (float): Mean photon number per pulse
        """
        self.mu = mu
    
    def poisson_distribution(self, n_max=20):
        """
        Calculate the photon number distribution P(n) up to n_max using Poisson distribution.
        
        Returns:
            np.array: Probability distribution of photon numbers
        """
        n_values = np.arange(n_max + 1)
        probs = poisson.pmf(n_values, self.mu)
        return probs


class Channel:
    """
    Represents the quantum channel between Alice and Bob.
    """
    def __init__(self, base_efficiency=0.8, distance=0, attenuation=0.2):
        """
        Initialize the channel with distance-dependent efficiency.
        
        Args:
            base_efficiency (float): Base channel transmission efficiency without distance (0-1)
            distance (float): Channel distance in kilometers
            attenuation (float): Fiber attenuation coefficient in dB/km
        """
        self.base_efficiency = base_efficiency
        self.distance = distance
        self.attenuation = attenuation
        self.efficiency = self.calculate_efficiency()
    
    def calculate_efficiency(self):
        """
        Calculate the actual channel efficiency based on distance.
        
        Returns:
            float: Actual channel efficiency after distance attenuation
        """
        # Calculate attenuation in dB
        attenuation_db = self.distance * self.attenuation
        
        # Convert to transmission efficiency: 10^(-attenuation_db/10)
        distance_factor = 10**(-attenuation_db/10)
        
        # Total efficiency is base efficiency times distance factor
        return self.base_efficiency * distance_factor
    
    def update_distance(self, distance):
        """
        Update the channel distance and recalculate efficiency.
        
        Args:
            distance (float): New channel distance in kilometers
        """
        self.distance = distance
        self.efficiency = self.calculate_efficiency()
    
    def transmit_pulse(self, is_non_empty):
        """
        Simulates the transmission of a pulse through the channel.
        
        Args:
            is_non_empty (bool): True if the pulse contains photons, False for empty pulse
            
        Returns:
            bool: True if the pulse is detected, False otherwise
        """
        if not is_non_empty:
            return False
        
        # Probability of detection is based on the channel efficiency
        return np.random.random() < self.efficiency


class Detector:
    """
    Represents a single-photon detector with noise characteristics.
    """
    def __init__(self, efficiency=0.2, dark_count_rate=500, time_window=1e-9):
        """
        Initialize detector with its characteristics.
        
        Args:
            efficiency (float): Detector efficiency (0-1)
            dark_count_rate (float): Dark count rate in counts per second
            time_window (float): Detection time window in seconds
        """
        self.efficiency = efficiency
        self.dark_count_rate = dark_count_rate
        self.time_window = time_window
        self.p_dark = 1 - np.exp(-dark_count_rate * time_window)
    
    def detect(self, received_pulse):
        """
        Simulates the detection process with detector efficiency and dark counts.
        
        Args:
            received_pulse (bool): True if pulse is received at the detector, False otherwise
            
        Returns:
            bool: True if detector clicks, False otherwise
        """
        # Check for dark count
        dark_count = np.random.random() < self.p_dark
        
        # Check for detection of received photon
        photon_detection = received_pulse and (np.random.random() < self.efficiency)
        
        # Detector clicks if either condition is met
        return dark_count or photon_detection


class COWProtocol:
    """
    Simulates the Coherent One-Way QKD protocol.
    """
    def __init__(self, mu=0.5, distance=0, 
                 detector_efficiency=0.2, 
                 dark_count_rate=500,
                 time_window=1e-9,
                 channel_base_efficiency=0.8,
                 attenuation=0.2,
                 data_line_ratio=0.9,  # 90% for data line, 10% for monitoring line
                 decoy_probability=0.1,  # Probability of sending a decoy sequence
                 repetition_rate=500e6):  # 500 MHz pulse repetition rate
        """
        Initialize the COW protocol simulator.
        
        Args:
            mu (float): Mean photon number
            distance (float): Distance between Alice and Bob in kilometers
            detector_efficiency (float): Bob's detector efficiency
            dark_count_rate (float): Dark count rate in counts per second
            time_window (float): Detection time window in seconds
            channel_base_efficiency (float): Base efficiency of channel
            attenuation (float): Fiber attenuation coefficient in dB/km
            data_line_ratio (float): Fraction of photons sent to data line (0-1)
            decoy_probability (float): Probability of sending a decoy sequence
            repetition_rate (float): Pulse repetition rate in Hz
        """
        self.source = PhotonSource(mu)
        self.mu = mu
        self.channel = Channel(channel_base_efficiency, distance, attenuation)
        self.data_detector = Detector(detector_efficiency, dark_count_rate, time_window)
        self.monitor_detector_1 = Detector(detector_efficiency, dark_count_rate, time_window)
        self.monitor_detector_2 = Detector(detector_efficiency, dark_count_rate, time_window)
        self.distance = distance
        self.data_line_ratio = data_line_ratio
        self.decoy_probability = decoy_probability
        self.repetition_rate = repetition_rate
        
        # Pre-calculate some frequently used values
        self.interference_visibility = 0.98  # Interferometer visibility
        
        # Parameters for error correction and privacy amplification
        self.error_correction_efficiency = 1.2  # Practical error correction overhead
        self.security_parameter = 0.5  # Monitoring line parameter
    
    def update_distance(self, distance):
        """
        Update the distance and recalculate channel efficiency.
        
        Args:
            distance (float): New distance in kilometers
        """
        self.distance = distance
        self.channel.update_distance(distance)
    
    def update_mu(self, mu):
        """
        Update the mean photon number.
        
        Args:
            mu (float): New mean photon number
        """
        self.mu = mu
        self.source = PhotonSource(mu)
    
    def _calculate_phase_qber(self):
        """
        Calculate the QBER component due to phase errors in COW protocol.
        This increases with distance due to imperfect visibility.
        
        Returns:
            float: Phase QBER component
        """
        # Apply distance-dependent visibility degradation
        distance_factor = 1 + 0.001 * self.distance  # Empirical visibility degradation
        effective_visibility = self.interference_visibility / distance_factor
        
        # Phase QBER based on visibility and mean photon number
        qber_phase = (1 - effective_visibility * np.exp(-self.mu/2)) / 2
        return qber_phase
    
    def calculate_qber(self):
        """
        Calculate the quantum bit error rate (QBER) for COW protocol.
        
        Returns:
            float: QBER as a percentage
        """
        # Calculate detection probabilities
        transmittance = self.channel.efficiency
        detection_prob = 1 - np.exp(-self.mu * transmittance * self.data_detector.efficiency)
        
        # Calculate bit error component (from dark counts)
        dark_count_prob = self.data_detector.p_dark
        qber_bit = dark_count_prob / (detection_prob + dark_count_prob) if (detection_prob + dark_count_prob) > 0 else 0.5
        
        # Calculate phase error component (from visibility and coherence)
        qber_phase = self._calculate_phase_qber()
        
        # Total QBER is a weighted combination of bit errors and phase errors
        qber_total = 0.5 * (qber_bit + qber_phase)
        
        # Convert to percentage
        return qber_total * 100
    
    def _h_binary(self, p):
        """
        Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p).
        
        Args:
            p (float): Probability (0 <= p <= 1)
            
        Returns:
            float: Binary entropy value
        """
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def calculate_skr(self):
        """
        Calculate the Secret Key Rate (SKR) for COW protocol.
        
        Returns:
            float: Secret key rate in bits per second
        """
        # Calculate detection probabilities
        transmittance = self.channel.efficiency
        detection_prob = 1 - np.exp(-self.mu * transmittance * self.data_detector.efficiency)
        
        # Calculate raw key rate
        # For low mu, detection probability is approximately linear with mu
        # For high mu, multi-photon events increase and reduce security
        raw_key_rate = self.repetition_rate * detection_prob * self.data_line_ratio * (1 - self.decoy_probability)
        
        # Calculate QBER (as a fraction, not percentage)
        qber = self.calculate_qber() / 100
        
        # If QBER is too high, no secure key is possible
        if qber >= 0.11:  # Security threshold for COW
            return 0
            
        # Multi-photon penalty factor - reduces security for high mu values
        # This factor creates the bell-shaped curve for SKR vs mu
        multi_photon_factor = np.exp(-2 * self.mu)  # Strongly penalize multi-photon events
        
        # Calculate secret key fraction after privacy amplification and error correction
        secret_fraction = (1 - self.security_parameter) * (1 - self.error_correction_efficiency * self._h_binary(qber))
        secret_fraction = secret_fraction * multi_photon_factor
        
        # Ensure the fraction is non-negative
        secret_fraction = max(0, secret_fraction)
        
        # Calculate final secret key rate
        skr = raw_key_rate * secret_fraction
        
        return skr


def plot_qber_vs_mu(mu_values=None, distance=50):
    """
    Plot QBER vs mean photon number μ for COW protocol.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        distance (float, optional): Distance in kilometers
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    qber_values = []
    
    # Create simulator
    simulator = COWProtocol(distance=distance)
    
    # Calculate QBER for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'b-', linewidth=2, label='QBER')
    plt.plot(mu_values, qber_values, 'bo', markersize=6)  # Add points for measurements
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    plt.title(f'Quantum Bit Error Rate vs Mean Photon Number', fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    return mu_values, qber_values


def plot_qber_vs_distance(distance_values=None, mu=0.5):
    """
    Plot QBER vs distance for COW protocol.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        mu (float, optional): Mean photon number
    """
    if distance_values is None:
        distance_values = np.linspace(0, 200, 20)
    
    qber_values = []
    
    # Create simulator
    simulator = COWProtocol(mu=mu)
    
    # Calculate QBER for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'r-', linewidth=2, label='QBER')
    plt.plot(distance_values, qber_values, 'ro', markersize=6)  # Add points for measurements
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    plt.title(f'Quantum Bit Error Rate vs Distance', fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    return distance_values, qber_values


def plot_skr_vs_mu(mu_values=None, distance=50):
    """
    Plot Secret Key Rate vs mean photon number μ for COW protocol.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        distance (float, optional): Distance in kilometers
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    skr_values = []
    
    # Create simulator
    simulator = COWProtocol(distance=distance)
    
    # Calculate SKR for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'g-', linewidth=2, label='SKR')
    plt.plot(mu_values, skr_values, 'go', markersize=6)  # Add points for measurements
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=20)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=20)
    plt.title(f'Secret Key Rate vs Mean Photon Number', fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    return mu_values, skr_values


def plot_skr_vs_distance(distance_values=None, mu=0.3):
    """
    Plot Secret Key Rate vs distance for COW protocol.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        mu (float, optional): Mean photon number
    """
    if distance_values is None:
        distance_values = np.linspace(0, 200, 20)
    
    skr_values = []
    
    # Create simulator
    simulator = COWProtocol(mu=mu)
    
    # Calculate SKR for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'm-', linewidth=2, label='SKR')
    plt.plot(distance_values, skr_values, 'mo', markersize=6)  # Add points for measurements
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=20)
    plt.title(f'Secret Key Rate vs Distance', fontsize=22)
    
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    return distance_values, skr_values
