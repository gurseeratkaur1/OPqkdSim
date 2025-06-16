import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import math

class WeakCoherentSource:
    """
    Simulates a weak coherent photon source for BB84 protocol.
    """
    def __init__(self, mu):
        """
        Initialize the photon source with mean photon number mu.
        
        Args:
            mu (float): Mean photon number per pulse
        """
        self.mu = mu
    
    def photon_distribution(self, n_max=20):
        """
        Calculate the Poisson photon number distribution for weak coherent states.
        
        Returns:
            np.array: Probability distribution of photon numbers
        """
        n_values = np.arange(n_max + 1)
        # Poisson distribution: P(n) = e^(-μ) * μ^n / n!
        p_n = np.exp(-self.mu) * (self.mu**n_values) / np.array([math.factorial(n) for n in n_values])
        return p_n


class Channel:
    """
    Represents the quantum channel between Alice and Bob.
    """
    def __init__(self, base_efficiency, distance=0, attenuation=0.2):
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
        
        # Optical misalignment that increases with distance
        self.misalignment_base = 0.015  # 1.5% base misalignment error
        self.misalignment_factor = 0.0002  # Increase per km
    
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
    
    def calculate_misalignment_error(self):
        """
        Calculate optical misalignment error based on distance.
        
        Returns:
            float: Misalignment error probability (0-1)
        """
        # Error increases with distance but saturates
        return min(0.1, self.misalignment_base + self.misalignment_factor * self.distance)


class Detector:
    """
    Represents a single-photon detector with noise characteristics.
    """
    def __init__(self, efficiency, dark_count_rate, time_window):
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
        
        # Detector afterpulsing probability
        self.afterpulsing_prob = 0.02
        
        # Detector timing jitter (as error probability)
        self.timing_jitter_error = 0.01
    
    def detect_probability(self, photons):
        """
        Calculate the probability of detection given number of photons.
        
        Args:
            photons (int): Number of photons arriving at detector
            
        Returns:
            float: Probability of detection
        """
        # Probability of at least one photon being detected
        if photons > 0:
            # 1 - probability that none are detected
            p_detect_signal = 1 - (1 - self.efficiency)**photons
            
            # Add saturation effect for multiple photons (models crosstalk and other non-linearities)
            saturation_factor = 1.0
            if photons > 1:
                # Detector saturation for multi-photon pulses
                saturation_factor = 1.0 + 0.02 * (photons - 1)
                
            return min(1.0, p_detect_signal * saturation_factor)
        return 0
    
    def dark_count_probability(self):
        """
        Calculate the probability of a dark count in the detection window.
        
        Returns:
            float: Dark count probability
        """
        return self.p_dark


class BB84Simulator:
    """
    Simulates the BB84 QKD protocol with weak coherent source.
    """
    def __init__(self, mu, detector_efficiency, channel_base_efficiency,
                 dark_count_rate, time_window, distance=0, attenuation=0.2, p_eve=0.0):
        """
        Initialize the BB84 simulator.
        
        Args:
            mu (float): Mean photon number
            detector_efficiency (float): Bob's detector efficiency
            channel_base_efficiency (float): Base efficiency of quantum channel
            dark_count_rate (float): Dark count rate in counts per second
            time_window (float): Detection time window in seconds
            distance (float): Distance between Alice and Bob in kilometers
            attenuation (float): Fiber attenuation coefficient in dB/km
            p_eve (float): Probability of Eve performing intercept-resend attack
        """
        self.source = WeakCoherentSource(mu)
        self.mu = mu
        self.channel = Channel(channel_base_efficiency, distance, attenuation)
        self.detector = Detector(detector_efficiency, dark_count_rate, time_window)
        self.distance = distance
        self.attenuation = attenuation
        self.p_eve = p_eve
        self.n_max = 10  # Maximum photon number to consider in calculations
        self.confidence = 0.95  # Statistical confidence for error estimation
        self.time_window = time_window
        self.repetition_rate = 1e6  # Default pulse rate: 1 MHz
        
        # Additional error sources
        self.optical_error_base = 0.01  # Base optical error rate (1%)
        self.multi_photon_error_factor = 0.02  # Additional error per photon for multi-photon pulses
    
    def update_distance(self, distance):
        """
        Update the distance between Alice and Bob and recalculate channel efficiency.
        
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
        self.source = WeakCoherentSource(mu)
    
    def calculate_raw_key_rate(self):
        """
        Calculate the raw key rate (before sifting).
        
        Returns:
            float: Raw key rate in bits per pulse
        """
        # Probability that at least one photon is received and detected
        p_distribution = self.source.photon_distribution(self.n_max)
        
        # Raw key rate before sifting - probability of detection per pulse
        raw_rate = 0
        for n in range(1, self.n_max + 1):  # Start from n=1 (at least one photon)
            # Probability that n photons are sent
            p_n = p_distribution[n]
            
            # Calculate probability of receiving and detecting at least one photon
            p_detect_n = 0
            for k in range(1, n + 1):  # k photons reach Bob
                # Probability that k out of n photons reach Bob
                p_trans = self.channel.transmission_probability(n, k)
                # Probability that at least one of k photons is detected
                p_detect = self.detector.detect_probability(k)
                p_detect_n += p_trans * p_detect
            
            raw_rate += p_n * p_detect_n
        
        # Add probability of dark count when no photon is detected
        p_no_photon = p_distribution[0]  # Probability of sending zero photons
        p_dark = self.detector.dark_count_probability()
        raw_rate += p_no_photon * p_dark
        
        # For non-zero photons that aren't detected, there's still a chance of dark count
        for n in range(1, self.n_max + 1):
            p_n = p_distribution[n]
            
            # Probability no photons are detected (includes all transmission possibilities)
            p_no_detect = 0
            for k in range(n + 1):  # 0 to n photons reach Bob
                p_trans = self.channel.transmission_probability(n, k)
                p_not_detect = (1 - self.detector.detect_probability(k))
                p_no_detect += p_trans * p_not_detect
            
            # Add probability of dark count when signal photons aren't detected
            raw_rate += p_n * p_no_detect * p_dark
        
        return raw_rate
    
    def calculate_sifted_key_rate(self):
        """
        Calculate the sifted key rate (after basis reconciliation).
        
        Returns:
            float: Sifted key rate in bits per pulse
        """
        # After basis reconciliation, approximately half of the raw bits remain
        return self.calculate_raw_key_rate() * 0.5
    
    def calculate_quantum_bit_error_rate(self):
        """
        Calculate the quantum bit error rate (QBER) for the BB84 protocol.
        
        Returns:
            float: QBER as a percentage
        """
        p_distribution = self.source.photon_distribution(self.n_max)
        
        # Channel efficiency decreases exponentially with distance
        channel_efficiency = self.channel.efficiency
        
        # Misalignment error inversely proportional to channel efficiency
        # (harder to maintain alignment as signal weakens)
        misalignment_error = self.channel.misalignment_base + self.channel.misalignment_factor * self.distance
        
        # Calculate error sources
        p_dark = self.detector.dark_count_probability()
        p_sig_correct = 0
        p_sig_error = 0
        
        # Calculate signal error probabilities
        for n in range(1, self.n_max + 1):
            p_n = p_distribution[n]
            multi_photon_error = self.optical_error_base
            if n > 1:
                multi_photon_error += self.multi_photon_error_factor * (n - 1)
            
            for k in range(1, n + 1):
                p_trans = self.channel.transmission_probability(n, k)
                p_detect = self.detector.detect_probability(k)
                
                # Total optical error without artificial caps
                p_optical_error = misalignment_error + self.detector.timing_jitter_error + multi_photon_error
                
                if self.p_eve == 0:
                    p_sig_correct += p_n * p_trans * p_detect * (1 - p_optical_error)
                    p_sig_error += p_n * p_trans * p_detect * p_optical_error
                else:
                    eve_no_error = (1 - self.p_eve) * (1 - p_optical_error) + self.p_eve * 0.75 * (1 - p_optical_error)
                    eve_error = (1 - self.p_eve) * p_optical_error + self.p_eve * 0.25 + self.p_eve * 0.75 * p_optical_error
                    
                    p_sig_correct += p_n * p_trans * p_detect * eve_no_error
                    p_sig_error += p_n * p_trans * p_detect * eve_error
        
        # Dark count errors (50% chance of wrong bit)
        p_dark_error = 0.5 * p_dark * (1 - p_sig_correct - p_sig_error)
        
        # Afterpulsing errors
        p_afterpulse_error = 0.5 * (p_sig_correct + p_sig_error) * self.detector.afterpulsing_prob
        
        # Total error probability
        p_error = p_sig_error + p_dark_error + p_afterpulse_error
        
        # QBER calculation - no artificial scaling needed
        p_detect_total = self.calculate_sifted_key_rate() * 2
        
        # As channel efficiency decreases, dark count contribution naturally increases
        qber = (p_error / p_detect_total) * 100 if p_detect_total > 0 else 100

        return qber

    def statistical_error(self, measured_error_rate, n_samples):
        """
        Calculate the statistical error in the QBER estimation.
        
        Args:
            measured_error_rate (float): Measured error rate (ζ)
            n_samples (int): Number of sample bits used for estimation
            
        Returns:
            float: Statistical error (ε)
        """
        if measured_error_rate == 0 or measured_error_rate == 1:
            return 0
        
        # ε = √[ζ(1-ζ)/(ζNSamples(1-S))]
        epsilon = np.sqrt(measured_error_rate * (1 - measured_error_rate) / 
                        (measured_error_rate * n_samples * (1 - self.confidence)))
        
        return epsilon
    
    def error_correction_efficiency(self, error_rate):
        """
        Calculate the fraction of bits lost due to error correction.
        
        Args:
            error_rate (float): Error rate (δ)
            
        Returns:
            float: Fraction of bits lost in error correction
        """
        if error_rate <= 0:
            return 0
        
        # r_ec = 1.1 × h_binary(error_rate)
        r_ec = 1.1 * self.h_binary(error_rate)
        
        return r_ec
    
    def privacy_amplification_efficiency(self, error_rate):
        """
        Calculate the fraction of bits lost due to privacy amplification.
        
        Args:
            error_rate (float): Error rate (δ)
            
        Returns:
            float: Fraction of bits lost in privacy amplification
        """
        # Simplified privacy amplification factor based on multi-photon probability
        p_distribution = self.source.photon_distribution(self.n_max)
        p_multi = sum(p_distribution[2:])  # Probability of multi-photon pulses
        
        # Base privacy amplification fraction
        r_pa_base = 0.1 + error_rate
        
        # Additional privacy amplification needed for multi-photon pulses
        r_pa_multi = p_multi * 0.5
        
        return r_pa_base + r_pa_multi
    
    def h_binary(self, p):
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
        Calculate the secret key rate (SKR) after error correction and privacy amplification.
        
        Returns:
            float: Secret key rate in bits per pulse
        """
        # Get photon number distribution
        p_distribution = self.source.photon_distribution(self.n_max)
        
        # Probability of single-photon pulses - crucial for secure key generation
        p_single = p_distribution[1]
        
        # Probability of multi-photon pulses - vulnerable to photon-number-splitting attacks
        p_multi = sum(p_distribution[2:])
        
        # Calculate the sifted key rate
        sifted_rate = self.calculate_sifted_key_rate()
        
        # Calculate QBER
        qber = self.calculate_quantum_bit_error_rate() / 100  # Convert from percentage to fraction
        
        # # If QBER too high, no secure key possible
        # if qber >= 0.11:
        #     return 0
        
        # Calculate transmission gain for single-photon pulses
        gain_single = 0
        for k in range(1, 2):  # For single-photon pulses, only k=1 is possible
            p_trans = self.channel.transmission_probability(1, k)
            p_detect = self.detector.detect_probability(k)
            gain_single += p_single * p_trans * p_detect
        
        # Calculate gain for multi-photon pulses (vulnerable to PNS attacks)
        gain_multi = 0
        for n in range(2, self.n_max + 1):  # Multi-photon pulses
            p_n = p_distribution[n]
            gain_n = 0
            for k in range(1, n + 1):  # k photons reach Bob
                p_trans = self.channel.transmission_probability(n, k)
                p_detect = self.detector.detect_probability(k)
                gain_n += p_trans * p_detect
            gain_multi += p_n * gain_n
        
        # Calculate detection probability from dark counts
        p_dark = self.detector.dark_count_probability()
        p_no_photon = p_distribution[0]  # Probability of sending zero photons
        gain_dark = p_no_photon * p_dark
        
        # Total gain (detection probability)
        gain_total = gain_single + gain_multi + gain_dark
        
        # Fraction of detections from single-photon pulses (secure)
        f_single = gain_single / gain_total if gain_total > 0 else 0
        
        # Fraction of detections from multi-photon pulses (vulnerable)
        f_multi = gain_multi / gain_total if gain_total > 0 else 0
        
        # Fraction of detections from dark counts (adds to error)
        f_dark = gain_dark / gain_total if gain_total > 0 else 0
        
        # Error correction and privacy amplification overhead
        r_ec = self.error_correction_efficiency(qber)
        
        # Modified QBER to account for relative contributions
        # Dark counts and multi-photon pulses add to effective error rate
        effective_qber = qber + 0.5 * f_dark + 0.1 * f_multi
        
        # Information leakage due to multi-photon pulses
        info_leakage = f_multi
        
        # Calculate the final secret key rate using the GLLP formula
        # R = R_sifted * [1 - h_2(QBER) - leakage]
        skr = sifted_rate * (1 - self.h_binary(effective_qber) - r_ec - info_leakage)
        
        # No negative key rates
        return max(0, skr)


def plot_qber_vs_mu(mu_values=None, time_window=10e-9, distance=50,
                   detector_efficiency=0.15, channel_base_efficiency=0.60, 
                   dark_count_rate=2000, p_eve=0.0):
    """
    Plot QBER vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float, optional): Distance in kilometers
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        p_eve (float, optional): Probability of eavesdropping
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 2.0, 40)
    
    qber_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            p_eve=p_eve
        )
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=5, color='magenta', linestyle='--')
    plt.axhline(y=11, color='red', linestyle='--')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Mean Photon Number')
    #plt.savefig('qber_vs_mu_bb84.png')
    plt.show()
    
    return mu_values, qber_values


def plot_skr_vs_mu(mu_values=None, time_window=10e-9, distance=50,
                  detector_efficiency=0.15, channel_base_efficiency=0.60, 
                  dark_count_rate=2000, repetition_rate=1000000, p_eve=0.0):
    """
    Plot Secret Key Rate vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float, optional): Distance in kilometers
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        repetition_rate (float, optional): Source repetition rate in Hz
        p_eve (float, optional): Probability of eavesdropping
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.2, 40)
    
    skr_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            p_eve=p_eve
        )
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title('Secret Key Rate vs Mean Photon Number')
    #plt.savefig('skr_vs_mu_bb84.png')
    plt.show()
    
    return mu_values, skr_values


def plot_qber_vs_distance(distance_values=None, time_window=10e-9, mu=0.5,
                         detector_efficiency=0.15, channel_base_efficiency=0.60, 
                         dark_count_rate=2000, p_eve=0.0):
    """
    Plot QBER vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        mu (float, optional): Mean photon number
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        p_eve (float, optional): Probability of eavesdropping
    """
    if distance_values is None:
        distance_values = np.linspace(0, 120, 40)
    
    qber_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        detector_efficiency=detector_efficiency,
        channel_base_efficiency=channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        p_eve=p_eve
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'ro-', linewidth=2)
    plt.grid(True)
    plt.axhline(y=5, color='magenta', linestyle='--')
    plt.axhline(y=11, color='red', linestyle='--')
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Distance')
    #plt.savefig('qber_vs_distance_bb84.png')
    plt.show()
    
    return distance_values, qber_values


def plot_skr_vs_distance(distance_values=None, time_window=10e-9, mu=0.1,
                        detector_efficiency=0.15, channel_base_efficiency=0.60, 
                        dark_count_rate=2000, repetition_rate=1000000, p_eve=0.0):
    """
    Plot Secret Key Rate vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        mu (float, optional): Mean photon number
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        repetition_rate (float, optional): Source repetition rate in Hz
        p_eve (float, optional): Probability of eavesdropping
    """
    if distance_values is None:
        distance_values = np.linspace(0, 120, 40)
    
    skr_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        detector_efficiency=detector_efficiency,
        channel_base_efficiency=channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        p_eve=p_eve
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(distance_values, skr_values, 'mo-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title('Secret Key Rate vs Distance')
    #plt.savefig('skr_vs_distance_bb84.png')
    plt.show()
    
    return distance_values, skr_values


def plot_qber_skr_vs_eavesdropping(p_eve_values=None, time_window=10e-9, distance=50, mu=0.1,
                                  detector_efficiency=0.15, channel_base_efficiency=0.60, 
                                  dark_count_rate=2000, repetition_rate=1000000):
    """
    Plot QBER and SKR vs eavesdropping probability (p_eve).
    
    Args:
        p_eve_values (list, optional): List of eavesdropping probability values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float, optional): Distance in kilometers
        mu (float, optional): Mean photon number
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        repetition_rate (float, optional): Source repetition rate in Hz
    """
    if p_eve_values is None:
        p_eve_values = np.linspace(0, 0.5, 50)  # From 0% to 100% eavesdropping
    
    qber_values = []
    skr_values = []
    
    for p_eve in p_eve_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            p_eve=p_eve
        )
        
        # Calculate QBER
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
        
        # Calculate SKR
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    # Plot QBER vs p_eve
    plt.figure(figsize=(10, 6))
    plt.plot(p_eve_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=11, color='red', linestyle='--', label='Security Threshold (11%)')
    plt.grid(True)
    plt.xlabel('Eavesdropping Probability (p_eve)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Eavesdropping Probability')
    plt.legend()
    plt.show()
    
    # Plot SKR vs p_eve
    plt.figure(figsize=(10, 6))
    plt.plot(p_eve_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Eavesdropping Probability (p_eve)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title('Secret Key Rate vs Eavesdropping Probability')
    plt.show()
    
    return p_eve_values, qber_values, skr_values