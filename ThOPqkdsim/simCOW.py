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
    Includes both fiber and FSO (Free Space Optical) channel modeling options.
    """
    def __init__(self, base_efficiency, distance=0, attenuation=0.2, mode="fiber"):
        """
        Initialize the channel with distance-dependent efficiency.
        
        Args:
            base_efficiency (float): Base channel transmission efficiency without distance (0-1)
            distance (float): Channel distance in kilometers
            attenuation (float): Fiber attenuation coefficient in dB/km
            mode (str): Channel mode - "fiber" or "fso"
        """
        self.base_efficiency = base_efficiency
        self.distance = distance
        self.attenuation = attenuation
        
        # FSO specific parameters with default values
        self.transmitter_efficiency = 0.9  # Efficiency of transmitter optics
        self.receiver_efficiency = 0.9     # Efficiency of receiver optics
        self.transmitter_diameter = 0.1    # Diameter of transmitter aperture in meters
        self.receiver_diameter = 0.3       # Diameter of receiver aperture in meters
        self.beam_divergence = 0.001       # Beam divergence angle in radians (1 mrad)
        self.wavelength = 850e-9           # Wavelength in meters (850 nm)
        self.pointing_error = 1e-6         # Pointing error in radians
        
        # Optical misalignment that increases with distance
        self.misalignment_base = 0.015     # 1.5% base misalignment error
        self.misalignment_factor = 0.0002  # Increase per km
        
        # Set mode and calculate efficiency
        self.mode = mode
        self.efficiency = self.calculate_efficiency()
    
    def calculate_efficiency(self):
        """
        Calculate the actual channel efficiency based on distance and mode.
        
        Returns:
            float: Actual channel efficiency after distance attenuation
        """
        if self.mode == "fiber":
            return self._calculate_fiber_efficiency()
        elif self.mode == "fso":
            return self._calculate_fso_efficiency()
        else:
            raise ValueError(f"Unknown channel mode: {self.mode}")
    
    def _calculate_fiber_efficiency(self):
        """
        Calculate efficiency for fiber optic channel.
        
        Returns:
            float: Channel efficiency for fiber
        """
        # Calculate attenuation in dB
        attenuation_db = self.distance * self.attenuation
        
        # Convert to transmission efficiency: 10^(-attenuation_db/10)
        distance_factor = 10**(-attenuation_db/10)
        
        # Total efficiency is base efficiency times distance factor
        return self.base_efficiency * distance_factor
    
    def _calculate_fso_efficiency(self):
        """
        Calculate efficiency for FSO channel based on provided model.
        
        Returns:
            float: Channel efficiency for FSO
        """
        # For zero distance, return direct efficiency without atmospheric effects
        if self.distance <= 1e-6:  # Effectively zero
            return self.base_efficiency * self.transmitter_efficiency * self.receiver_efficiency
    
        # Calculate geometrical loss factor
        beam_diameter_at_receiver = self.transmitter_diameter + (self.distance * 1000 * self.beam_divergence)
        geo_factor = min(1.0, (self.receiver_diameter / beam_diameter_at_receiver)**2)
        
        # Calculate simplified turbulence-induced scintillation loss
        # Using a simplified model based on distance
        turb_factor = np.exp(-0.05 * self.distance)  # Simplified exponential decay with distance
        
        # Calculate simplified beam wandering effect
        # Increases with distance
        pointing_variance = (self.pointing_error * self.distance * 1000)**2
        beam_spot_size = (self.beam_divergence * self.distance * 1000 / 2)**2
        bw_factor = np.exp(-2 * pointing_variance / beam_spot_size)
        
        # Calculate overall transmission efficiency
        total_efficiency = (self.base_efficiency * geo_factor * self.transmitter_efficiency * 
                            self.receiver_efficiency * turb_factor * bw_factor)
        
        return min(1.0, max(0.0, total_efficiency))  # Ensure efficiency is between 0 and 1
    
    def update_distance(self, distance):
        """
        Update the channel distance and recalculate efficiency.
        
        Args:
            distance (float): New channel distance in kilometers
        """
        self.distance = distance
        self.efficiency = self.calculate_efficiency()
    
    def update_mode(self, mode):
        """
        Update the channel mode and recalculate efficiency.
        Default FSO parameters are automatically used when switching to FSO mode.
        
        Args:
            mode (str): New channel mode ("fiber" or "fso")
        """
        if mode not in ["fiber", "fso"]:
            raise ValueError(f"Unsupported channel mode: {mode}. Use 'fiber' or 'fso'.")
            
        self.mode = mode
        self.efficiency = self.calculate_efficiency()
    
    def set_fso_parameters(self, transmitter_diameter=None, receiver_diameter=None, 
                          beam_divergence=None, wavelength=None, pointing_error=None,
                          transmitter_efficiency=None, receiver_efficiency=None):
        """
        Update FSO-specific parameters. Only updates the parameters that are provided.
        
        Args:
            transmitter_diameter (float, optional): Diameter of transmitter aperture in meters
            receiver_diameter (float, optional): Diameter of receiver aperture in meters
            beam_divergence (float, optional): Beam divergence angle in radians
            wavelength (float, optional): Wavelength in meters
            pointing_error (float, optional): Pointing error in radians
            transmitter_efficiency (float, optional): Efficiency of transmitter optics
            receiver_efficiency (float, optional): Efficiency of receiver optics
        """
        if transmitter_diameter is not None:
            self.transmitter_diameter = transmitter_diameter
        if receiver_diameter is not None:
            self.receiver_diameter = receiver_diameter
        if beam_divergence is not None:
            self.beam_divergence = beam_divergence
        if wavelength is not None:
            self.wavelength = wavelength
        if pointing_error is not None:
            self.pointing_error = pointing_error
        if transmitter_efficiency is not None:
            self.transmitter_efficiency = transmitter_efficiency
        if receiver_efficiency is not None:
            self.receiver_efficiency = receiver_efficiency
            
        # Recalculate efficiency if in FSO mode
        if self.mode == "fso":
            self.efficiency = self.calculate_efficiency()
    
    def transmission_probability(self, sent_photons, received_photons):
        """
        Calculate probability of receiving photons given sent photons.
        
        Args:
            sent_photons (int): Number of photons sent
            received_photons (int): Number of photons received
            
        Returns:
            float: Probability of receiving the specified number of photons
        """
        if received_photons > sent_photons:
            return 0.0
        
        return binom.pmf(received_photons, sent_photons, self.efficiency)
    
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
                 channel_mode="fiber",  # New parameter for channel mode
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
            channel_mode (str): Channel mode - "fiber" or "fso"
            data_line_ratio (float): Fraction of photons sent to data line (0-1)
            decoy_probability (float): Probability of sending a decoy sequence
            repetition_rate (float): Pulse repetition rate in Hz
        """
        self.source = PhotonSource(mu)
        self.mu = mu
        self.channel = Channel(channel_base_efficiency, distance, attenuation, channel_mode)
        self.data_detector = Detector(detector_efficiency, dark_count_rate, time_window)
        self.monitor_detector_1 = Detector(detector_efficiency, dark_count_rate, time_window)
        self.monitor_detector_2 = Detector(detector_efficiency, dark_count_rate, time_window)
        self.distance = distance
        self.data_line_ratio = data_line_ratio
        self.decoy_probability = decoy_probability
        self.repetition_rate = repetition_rate
        self.channel_mode = channel_mode
        
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
    
    def update_channel_mode(self, mode):
        """
        Update the channel mode between fiber and FSO.
        
        Args:
            mode (str): New channel mode ("fiber" or "fso")
        """
        if mode not in ["fiber", "fso"]:
            raise ValueError(f"Unsupported channel mode: {mode}. Use 'fiber' or 'fso'.")
        
        self.channel_mode = mode
        self.channel.update_mode(mode)
    
    def set_fso_parameters(self, **kwargs):
        """
        Set parameters specific to FSO channel.
        
        Args:
            **kwargs: Key-value pairs of FSO parameters to update
                Possible keys: transmitter_diameter, receiver_diameter, beam_divergence,
                wavelength, pointing_error, transmitter_efficiency, receiver_efficiency
        """
        self.channel.set_fso_parameters(**kwargs)
    
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
    


def plot_qber_vs_mu(mu_values=None, distance=50, channel_mode="fiber"):
    """
    Plot QBER vs mean photon number μ for COW protocol.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        distance (float, optional): Distance in kilometers
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    qber_values = []
    
    # Create simulator with specified channel mode
    simulator = COWProtocol(distance=distance, channel_mode=channel_mode)
    
    # Calculate QBER for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'b-', linewidth=2, label=f'QBER ({channel_mode.upper()})')
    plt.plot(mu_values, qber_values, 'bo', markersize=6)  # Add points for measurements
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    plt.title(f'Quantum Bit Error Rate vs Mean Photon Number ({channel_mode.upper()}, {distance} km)', fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    return mu_values, qber_values


def plot_qber_vs_distance(distance_values=None, mu=0.5, channel_mode="fiber"):
    """
    Plot QBER vs distance for COW protocol.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        mu (float, optional): Mean photon number
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
    """
    if distance_values is None:
        # For FSO, typically shorter distances are relevant
        if channel_mode == "fso":
            distance_values = np.linspace(0, 20, 20)  # Shorter range for FSO
        else:
            distance_values = np.linspace(0, 200, 20)  # Longer range for fiber
    
    qber_values = []
    
    # Create simulator with specified channel mode
    simulator = COWProtocol(mu=mu, channel_mode=channel_mode)
    
    # Calculate QBER for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'r-', linewidth=2, label=f'QBER ({channel_mode.upper()})')
    plt.plot(distance_values, qber_values, 'ro', markersize=6)  # Add points for measurements
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    plt.title(f'Quantum Bit Error Rate vs Distance ({channel_mode.upper()}, μ={mu})', fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    return distance_values, qber_values


def plot_skr_vs_mu(mu_values=None, distance=50, channel_mode="fiber"):
    """
    Plot Secret Key Rate vs mean photon number μ for COW protocol.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        distance (float, optional): Distance in kilometers
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    skr_values = []
    
    # Create simulator with specified channel mode
    simulator = COWProtocol(distance=distance, channel_mode=channel_mode)
    
    # Calculate SKR for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'g-', linewidth=2, label=f'SKR ({channel_mode.upper()})')
    plt.plot(mu_values, skr_values, 'go', markersize=6)  # Add points for measurements
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=20)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=20)
    plt.title(f'Secret Key Rate vs Mean Photon Number ({channel_mode.upper()}, {distance} km)', fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)    
    plt.show()
    
    return mu_values, skr_values


def plot_skr_vs_distance(distance_values=None, mu=0.3, channel_mode="fiber"):
    """
    Plot Secret Key Rate vs distance for COW protocol.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        mu (float, optional): Mean photon number
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
    """
    if distance_values is None:
        # For FSO, typically shorter distances are relevant
        if channel_mode == "fso":
            distance_values = np.linspace(0, 20, 20)  # Shorter range for FSO
        else:
            distance_values = np.linspace(0, 200, 20)  # Longer range for fiber
    
    skr_values = []
    
    # Create simulator with specified channel mode
    simulator = COWProtocol(mu=mu, channel_mode=channel_mode)
    
    # Calculate SKR for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'm-', linewidth=2, label=f'SKR ({channel_mode.upper()})')
    plt.plot(distance_values, skr_values, 'mo', markersize=6)  # Add points for measurements
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=20)
    plt.title(f'Secret Key Rate vs Distance ({channel_mode.upper()}, μ={mu})', fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)    
    plt.show()
    
    return distance_values, skr_values


def compare_fiber_vs_fso(distance_values=None, mu=0.3, plot_type="skr"):
    """
    Compare fiber vs FSO performance on the same plot.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        mu (float, optional): Mean photon number
        plot_type (str, optional): Type of plot - "skr" or "qber"
    """
    if distance_values is None:
        # Use a range that works for both fiber and FSO for comparison
        distance_values = np.linspace(0, 15, 20)
    
    fiber_values = []
    fso_values = []
    
    # Create simulators for both modes
    fiber_simulator = COWProtocol(mu=mu, channel_mode="fiber")
    fso_simulator = COWProtocol(mu=mu, channel_mode="fso")
    
    # Calculate values for each distance
    for distance in distance_values:
        fiber_simulator.update_distance(distance)
        fso_simulator.update_distance(distance)
        
        if plot_type.lower() == "skr":
            fiber_value = fiber_simulator.calculate_skr()
            fso_value = fso_simulator.calculate_skr()
            y_label = "Secret Key Rate (bits/s)"
            title = f"Secret Key Rate Comparison: Fiber vs FSO (μ={mu})"
        else:  # qber
            fiber_value = fiber_simulator.calculate_qber()
            fso_value = fso_simulator.calculate_qber()
            y_label = "QBER (%)"
            title = f"QBER Comparison: Fiber vs FSO (μ={mu})"
        
        fiber_values.append(fiber_value)
        fso_values.append(fso_value)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, fiber_values, 'b-', linewidth=2, label='Fiber')
    plt.plot(distance_values, fiber_values, 'bo', markersize=6)
    plt.plot(distance_values, fso_values, 'r-', linewidth=2, label='FSO')
    plt.plot(distance_values, fso_values, 'ro', markersize=6)
    
    if plot_type.lower() == "qber":
        plt.axhline(y=11, color='k', linestyle='--', label='Security threshold (11%)')
    
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Use logarithmic scale for SKR when values span multiple orders of magnitude
    if plot_type.lower() == "skr" and max(fiber_values + fso_values) / (min(fiber_values + fso_values) + 1e-10) > 1000:
        plt.yscale('log')
    
    plt.show()
    
    return distance_values, fiber_values, fso_values


def plot_fso_parameter_impact(parameter, values, distance=5, mu=0.3, plot_type="skr"):
    """
    Plot the impact of various FSO parameters on performance.
    
    Args:
        parameter (str): FSO parameter to vary ('transmitter_diameter', 'receiver_diameter', 
                         'beam_divergence', 'pointing_error', etc.)
        values (list): List of parameter values to simulate
        distance (float): Fixed distance in kilometers
        mu (float): Fixed mean photon number
        plot_type (str): Type of plot - "skr" or "qber"
    """
    result_values = []
    
    # Create simulator
    simulator = COWProtocol(mu=mu, distance=distance, channel_mode="fso")
    
    # Calculate values for each parameter value
    for value in values:
        # Set the specified parameter
        simulator.set_fso_parameters(**{parameter: value})
        
        if plot_type.lower() == "skr":
            result = simulator.calculate_skr()
            y_label = "Secret Key Rate (bits/s)"
            title = f"Impact of {parameter.replace('_', ' ').title()} on Secret Key Rate"
        else:  # qber
            result = simulator.calculate_qber()
            y_label = "QBER (%)"
            title = f"Impact of {parameter.replace('_', ' ').title()} on QBER"
        
        result_values.append(result)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(values, result_values, 'g-', linewidth=2)
    plt.plot(values, result_values, 'go', markersize=6)
    
    if plot_type.lower() == "qber":
        plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    
    plt.grid(True, alpha=0.7)
    plt.xlabel(parameter.replace('_', ' ').title(), fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title(f"{title} (FSO, {distance} km, μ={mu})", fontsize=22)
    
    if plot_type.lower() == "qber":
        plt.legend(fontsize=18)
        
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)    
    plt.show()
    
    return values, result_values