import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


class PhotonSource:
    """
    Simulates a Type-II SPDC photon source with photon number distributions.
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
        Calculate the photon number distribution PS(n) up to n_max.
        
        Returns:
            np.array: Probability distribution of photon numbers
        """
        n_values = np.arange(n_max + 1)
        p_s = (n_values + 1) * ((self.mu/2)**n_values) / ((1 + (self.mu/2))**(n_values + 2))
        return p_s


class Channel:
    """
    Represents the quantum channel between Alice and Bob.
    Includes both fiber and FSO (Free Space Optical) channel modeling options.
    """
    def __init__(self, base_efficiency, distance=0, attenuation=0.2, mode="fiber", atmos_attenuation=0.2,
                transmitter_diameter=0.1, receiver_diameter=0.3, beam_divergence=0.001,
                misalignment_base=0.015, misalignment_factor=0.0002):
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
        self.atmospheric_attenuation = atmos_attenuation    # Atmospheric attenuation in dB/km
        self.transmitter_efficiency = transmitter_diameter  # Efficiency of transmitter optics
        self.receiver_efficiency = receiver_diameter         # Efficiency of receiver optics
        self.transmitter_diameter = transmitter_diameter     # Diameter of transmitter aperture in meters
        self.receiver_diameter = receiver_diameter           # Diameter of receiver aperture in meters
        self.beam_divergence = beam_divergence       # Beam divergence angle in radians

        # Optical misalignment that increases with distance
        self.misalignment_base = misalignment_base           # 1.5% base misalignment error
        self.misalignment_factor = misalignment_factor       # Increase per km
        
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
            return self.base_efficiency 
    
        # Calculate geometrical loss factor
        beam_diameter_at_receiver = self.transmitter_diameter + (self.distance * 1000 * self.beam_divergence)
        geo_factor = min(1.0, (self.receiver_diameter / beam_diameter_at_receiver)**2)

        #calculate atmospheric loss factor
        atmos_loss = np.exp(-self.atmospheric_attenuation * self.distance)
        
    
        # Calculate overall transmission efficiency
        total_efficiency = (self.base_efficiency * geo_factor * atmos_loss)


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

    def set_fso_parameters(self, transmitter_diameter=None, receiver_diameter=None, atmos_attenuation=None,
                          beam_divergence=None):
        """
        Update FSO-specific parameters. Only updates the parameters that are provided.
        
        Args:
            transmitter_diameter (float, optional): Diameter of transmitter aperture in meters
            receiver_diameter (float, optional): Diameter of receiver aperture in meters
            beam_divergence (float, optional): Beam divergence angle in radians
        """
        if transmitter_diameter is not None:
            self.transmitter_diameter = transmitter_diameter
        if receiver_diameter is not None:
            self.receiver_diameter = receiver_diameter
        if beam_divergence is not None:
            self.beam_divergence = beam_divergence
        if atmos_attenuation is not None:
            self.atmospheric_attenuation = atmos_attenuation

            
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
        self.p_noise = 1 - np.exp(-dark_count_rate * time_window)
    
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
            return 1 - (1 - self.efficiency)**photons
        return 0


class BBM92Simulator:
    """
    Simulates the BBM92 QKD protocol with type-II SPDC source.
    """
    def __init__(self, mu, alice_detector_efficiency, bob_detector_efficiency, 
                 alice_channel_base_efficiency, bob_channel_base_efficiency,
                 dark_count_rate, time_window, distance=0, attenuation=0.2, 
                 p_eve=0.0, channel_mode="fiber", ec_eff_factor=1.1, e1_factor=1.05,
                 alice_atmos_attenuation=0.2, alice_transmitter_diameter=0.1, alice_receiver_diameter=0.3,
                 alice_misalignment_base=0.015, alice_misalignment_factor=0.0002, alice_beam_divergence=0.001,
                 bob_atmos_attenuation=0.2, bob_transmitter_diameter=0.1, bob_receiver_diameter=0.3,
                 bob_misalignment_base=0.015, bob_misalignment_factor=0.0002, bob_beam_divergence=0.001):
        """
        Initialize the BBM92 simulator.
        
        Args:
            mu (float): Mean photon number
            alice_detector_efficiency (float): Alice's detector efficiency
            bob_detector_efficiency (float): Bob's detector efficiency
            alice_channel_base_efficiency (float): Base efficiency of channel to Alice without distance
            bob_channel_base_efficiency (float): Base efficiency of channel to Bob without distance
            dark_count_rate (float): Dark count rate in counts per second
            time_window (float): Detection time window in seconds
            distance (float): Distance between Alice and Bob in kilometers
            attenuation (float): Fiber attenuation coefficient in dB/km
            p_eve (float): Probability of Eve performing intercept-resend attack
            channel_mode (str): Channel mode - "fiber" or "fso"
            ec_eff_factor (float): Efficiency factor for error correction
            e1_factor (float): error rate for privacy amplification calculation estimates how much 
                worse (or better) the error rate is on single-photon events compared to the total.
        """
        self.source = PhotonSource(mu)
        self.mu = mu
        self.alice_channel = Channel(alice_channel_base_efficiency, 0, attenuation,
                                     channel_mode, atmos_attenuation=alice_atmos_attenuation,
                                      transmitter_diameter=alice_transmitter_diameter, 
                                      receiver_diameter=alice_receiver_diameter,
                                     misalignment_base=alice_misalignment_base,
                                      misalignment_factor=alice_misalignment_factor,
                                      beam_divergence=alice_beam_divergence)  # Assuming source is at Alice
        self.bob_channel = Channel(bob_channel_base_efficiency, distance, attenuation, 
                                   channel_mode, atmos_attenuation=bob_atmos_attenuation,
                                    transmitter_diameter=bob_transmitter_diameter, 
                                    receiver_diameter=bob_receiver_diameter,
                                    misalignment_base=bob_misalignment_base,
                                    misalignment_factor=bob_misalignment_factor,
                                    beam_divergence=bob_beam_divergence)
        self.alice_detector = Detector(alice_detector_efficiency, dark_count_rate, time_window)
        self.bob_detector = Detector(bob_detector_efficiency, dark_count_rate, time_window)
        self.distance = distance
        self.attenuation = attenuation
        self.p_eve = p_eve
        self.channel_mode = channel_mode
        self.n_max = 10  # Maximum photon number to consider in calculations
        self.ec_eff_factor = ec_eff_factor  # Efficiency factor for error correction
        self.e1_factor = e1_factor  # Error rate factor for privacy amplification
    
    def update_distance(self, distance):
        """
        Update the distance between Alice and Bob and recalculate channel efficiencies.
        
        Args:
            distance (float): New distance in kilometers
        """
        self.distance = distance
        self.bob_channel.update_distance(distance)
        # Recalculate any other distance-dependent parameters here
    
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
        Update the channel mode (fiber or FSO) for both Alice and Bob channels.
        
        Args:
            mode (str): New channel mode ("fiber" or "fso")
        """
        self.channel_mode = mode
        self.alice_channel.update_mode(mode)
        self.bob_channel.update_mode(mode)
    
    def set_fso_parameters(self, transmitter_diameter=None, receiver_diameter=None, 
                          beam_divergence=None):
        """
        Update FSO-specific parameters for both Alice and Bob channels.
        
        Args:
            transmitter_diameter (float, optional): Diameter of transmitter aperture in meters
            receiver_diameter (float, optional): Diameter of receiver aperture in meters
            beam_divergence (float, optional): Beam divergence angle in radians
        """
        self.alice_channel.set_fso_parameters(
            transmitter_diameter, receiver_diameter, beam_divergence
        )
        self.bob_channel.set_fso_parameters(
            transmitter_diameter, receiver_diameter, beam_divergence
        )
    
    def calculate_joint_probabilities(self):
        """
        Calculate joint probabilities P(l,α,β) for all relevant photon numbers.
        Include detector efficiencies in the calculation.
        
        Returns:
            dict: Dictionary with (l,α,β) tuples as keys and probabilities as values
        """
        joint_probs = {}
        ps = self.source.photon_distribution(self.n_max)
        
        for l in range(self.n_max + 1):
            for alpha in range(l + 1):
                for beta in range(l + 1):
                    # Calculate channel transmission probabilities
                    p_alice_channel = self.alice_channel.transmission_probability(l, alpha)
                    p_bob_channel = self.bob_channel.transmission_probability(l, beta)
                    
                    # Calculate detection probabilities considering detector efficiency
                    p_alice_detect = self.alice_detector.detect_probability(alpha) 
                    p_bob_detect = self.bob_detector.detect_probability(beta)
                    
                    # Joint probability accounting for source, channel, and detector
                    joint_probs[(l, alpha, beta)] = p_alice_channel * p_bob_channel * ps[l] * p_alice_detect * p_bob_detect
        
        return joint_probs
    
    def calculate_p_rec(self):
        """
        Calculate the probability that both Alice and Bob's detectors record a click.
        
        Returns:
            float: Probability of recording a coincidence
        """
        joint_probs = self.calculate_joint_probabilities()
        p_an = self.alice_detector.p_noise
        p_bn = self.bob_detector.p_noise
        p_an4 = 1 - (1 - p_an)**4
        p_bn4 = 1 - (1 - p_bn)**4
        
        # Probability of signal-only detection (both receive and detect photons)
        p_both_signal = sum(prob for (l, alpha, beta), prob in joint_probs.items() 
                           if l > 0 and alpha > 0 and beta > 0)
        
        # Probability Alice gets noise, Bob gets signal
        p_alice_noise_bob_signal = sum(
            # Bob's photon detection probability * Alice's noise probability
            self.bob_detector.detect_probability(beta) * p_an4 * 
            # Channel transmission probabilities * source probability
            self.bob_channel.transmission_probability(l, beta) * 
            self.alice_channel.transmission_probability(l, 0) * 
            self.source.photon_distribution(self.n_max)[l]
            for l in range(1, self.n_max + 1)
            for beta in range(1, l + 1)
        )
        
        # Probability Alice gets signal, Bob gets noise
        p_alice_signal_bob_noise = sum(
            # Alice's photon detection probability * Bob's noise probability
            self.alice_detector.detect_probability(alpha) * p_bn4 * 
            # Channel transmission probabilities * source probability
            self.alice_channel.transmission_probability(l, alpha) * 
            self.bob_channel.transmission_probability(l, 0) * 
            self.source.photon_distribution(self.n_max)[l]
            for l in range(1, self.n_max + 1)
            for alpha in range(1, l + 1)
        )
        
        # Probability both get noise
        p_both_noise = p_an4 * p_bn4
        
        # Final coincidence probability
        p_rec = p_both_signal + p_alice_noise_bob_signal + p_alice_signal_bob_noise + p_both_noise
        
        return p_rec
    
    def calculate_p_key(self):
        """
        Calculate the probability that a bit is recorded and used in the key.
        
        Returns:
            float: Probability of recording a bit in the final key
        """
        p_rec = self.calculate_p_rec()
        p_key = 0.5 * p_rec  # Factor of 1/2 due to basis reconciliation
        
        return p_key
    
    def calculate_p_signal(self, joint_probs):
        """
        Calculate P(A=B, Signal) - probability that Alice and Bob's bits agree
        given that detectors detect photons.
        
        Args:
            joint_probs (dict): Joint probabilities P(l,α,β)
            
        Returns:
            float: Probability of bit agreement due to signal
        """
        p_signal = 0.0
        
        for (l, alpha, beta), prob in joint_probs.items():
            if l > 0 and alpha > 0 and beta > 0:
                # Determine P(A=B|l,α,β) based on the four cases
                if l == alpha and l == beta:
                    p_ab = (1/2)**(l-1)
                elif l == alpha and l != beta:
                    p_ab = (1/2)**beta
                elif l != alpha and l == beta:
                    p_ab = (1/2)**alpha
                else:  # l != alpha and l != beta
                    p_ab = (1/2)**(alpha + beta)
                
                p_signal += p_ab * prob
        
        return p_signal
    
    def calculate_p_noise(self):
        """
        Calculate P(A=B, Noise) - probability that Alice and Bob's bits agree
        when the clicks are only generated by noise.
        
        Returns:
            float: Probability of bit agreement due to noise
        """
        p_an4 = 1 - (1 - self.alice_detector.p_noise)**4
        p_bn4 = 1 - (1 - self.bob_detector.p_noise)**4
        
        # Factor of 1/16 = (1/4)^2 due to probability of choosing same basis and detector
        return (p_an4 * p_bn4) / 16
    
    def calculate_p_signal_noise(self, joint_probs):
        """
        Calculate P(A=B, Signal, Noise) - probability of receiving clicks from signal and noise.
        
        Args:
            joint_probs (dict): Joint probabilities P(l,α,β)
            
        Returns:
            float: Probability of bit agreement due to signal and noise
        """
        p_an = self.alice_detector.p_noise
        p_bn = self.bob_detector.p_noise
        p_an4 = 1 - (1 - p_an)**4
        p_bn4 = 1 - (1 - p_bn)**4
        p_signal_noise = 0.0
        
        for (l, alpha, beta), prob in joint_probs.items():
            if l > 0:
                # Cases where both receive photons
                if alpha > 0 and beta > 0:
                    if l == alpha and l == beta:
                        p_ab = (p_an4 + p_bn4) * (1/2)**(l+2)
                    elif l == alpha and l != beta:
                        p_ab = (p_an4 + p_bn4) * (1/2)**(beta+2)
                    elif l != alpha and l == beta:
                        p_ab = (p_an4 + p_bn4) * (1/2)**(alpha+2)
                    else:  # l != alpha and l != beta
                        p_ab = (p_an4 + p_bn4) * (1/2)**(alpha+beta+2)
                    p_signal_noise += p_ab * prob
                # Case where only Alice receives photons
                elif alpha > 0 and beta == 0:
                    p_ab = p_bn4 * (1/2)**(alpha+2)
                    p_signal_noise += p_ab * prob
                # Case where only Bob receives photons
                elif alpha == 0 and beta > 0:
                    p_ab = p_an4 * (1/2)**(beta+2)
                    p_signal_noise += p_ab * prob
        
        return p_signal_noise
    
    def calculate_p_signal_eir(self, joint_probs):
        """
        Calculate P(A=B, Signal, E.I.R) - probability that Alice and Bob's detectors agree
        when Eve performs the intercept-resend attack.
        
        Args:
            joint_probs (dict): Joint probabilities P(l,α,β)
            
        Returns:
            float: Probability of bit agreement with Eve's attack
        """
        if self.p_eve == 0:
            return 0.0
            
        p_signal_eir = 0.0
        
        for (l, alpha, beta), prob in joint_probs.items():
            if l > 0 and alpha > 0 and beta > 0:
                # Calculate P(E.I.R|β)
                p_eir_beta = 0.0
                for i in range(1, beta + 1):
                    p_eir_beta += binom.pmf(i, beta, self.p_eve)
                
                # Determine P(A=B|l,α,β) as in calculate_p_signal
                if l == alpha and l == beta:
                    p_ab = (1/2)**(l-1)
                elif l == alpha and l != beta:
                    p_ab = (1/2)**beta
                elif l != alpha and l == beta:
                    p_ab = (1/2)**alpha
                else:  # l != alpha and l != beta
                    p_ab = (1/2)**(alpha + beta)
                
                p_signal_eir += p_ab * prob * p_eir_beta
        
        return p_signal_eir
    
    def calculate_qber(self):
        """
        Calculate the quantum bit error rate (QBER).
        
        Returns:
            float: QBER as a percentage
        """
        joint_probs = self.calculate_joint_probabilities()
        p_signal = self.calculate_p_signal(joint_probs)
        p_noise = self.calculate_p_noise()
        p_signal_noise = self.calculate_p_signal_noise(joint_probs)
        p_signal_eir = self.calculate_p_signal_eir(joint_probs)
        
        p_rec = self.calculate_p_rec()
        p_key = self.calculate_p_key()
        
        # Calculate P(A=B, Key)
        p_ab_key = 0.5 * p_signal + p_noise - 0.5 * p_signal_noise - 0.5 * p_signal_eir
        
        # Calculate P(A=B|Key)
        p_cor = (2 * p_ab_key) / p_rec if p_rec > 0 else 0
        
        # Calculate BER (δ)
        qber = 0.5 * (1 - p_cor) * 100  # Convert to percentage
        
        return qber
    
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
    
    def error_correction_efficiency(self, error_rate):
        """
        Calculate the fraction of bits lost due to error correction.
        
        Args:
            error_rate (float): Error rate (δ)
            
        Returns:
            float: Fraction of bits lost in error correction
        """
        # if error_rate <= 0:
        #     return 0

        # r_ec = ec_eff_factor × h_binary(error_rate)
        r_ec = self.ec_eff_factor * self.h_binary(error_rate)

        return r_ec
    
    def calculate_skr(self, key_length):
        """
        Calculate the Secret Key Rate (SKR) after privacy amplification.
        
        Args:
            key_length (int): Length of the raw key (number of bits)
            
        Returns:
            float: Secret key rate in bits per second
        """
        p_key = self.calculate_p_key()
        qber = self.calculate_qber() / 100  # Convert from percentage to fraction
        
        # Use the QBER to estimate secret key fraction using privacy amplification
        # Formula based on BBM92 information theory (simplified)
        # if qber >= 0.11:  # No secure key possible if QBER too high
        #     return 0
        
        #privacy amplification factor
        info_leak = self.h_binary( self.e1_factor * qber)
        # Calculate secret key fraction using simplified asymptotic formula
        r = 1 - info_leak - self.error_correction_efficiency(qber)
        
        # Calculate final secret key rate
        skr = p_key * r   # for bits per channel use

        return max(0, skr)




def plot_qber_vs_mu(mu_values=None, time_window=1e-9, distance=10,
                    alice_detector_efficiency=0.20, bob_detector_efficiency=0.20,
                    alice_channel_base_efficiency=1.00, bob_channel_base_efficiency=0.1913, 
                    dark_count_rate=1000, channel_mode="fiber", fso_params=None):
    """
    Plot QBER vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float, optional): Distance in kilometers
        alice_detector_efficiency (float, optional): Alice's detector efficiency
        bob_detector_efficiency (float, optional): Bob's detector efficiency
        alice_channel_base_efficiency (float, optional): Base efficiency of Alice's channel
        bob_channel_base_efficiency (float, optional): Base efficiency of Bob's channel
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): FSO specific parameters dictionary
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    qber_values = []
    
    for mu in mu_values:
        simulator = BBM92Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            channel_mode=channel_mode
        )
        
        # Set FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'b', linewidth=3.5, label=f'QBER at {distance} km')
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    
    # channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    # plt.title(f'Quantum Bit Error Rate vs Mean Photon Number ({channel_type})', fontsize=22)
    
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
    return mu_values, qber_values

def plot_skr_vs_mu(mu_values=None, time_window=1e-9, key_length=1000000, distance=10,
                   alice_detector_efficiency=0.20, bob_detector_efficiency=0.20,
                   alice_channel_base_efficiency=1.00, bob_channel_base_efficiency=0.1913,
                   dark_count_rate=1000, repetition_rate=1000000, 
                   channel_mode="fiber", fso_params=None):
    """
    Plot Secret Key Rate vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        key_length (int, optional): Raw key length in bits
        distance (float, optional): Distance in kilometers
        alice_detector_efficiency (float, optional): Alice's detector efficiency
        bob_detector_efficiency (float, optional): Bob's detector efficiency
        alice_channel_base_efficiency (float, optional): Base efficiency of Alice's channel
        bob_channel_base_efficiency (float, optional): Base efficiency of Bob's channel
        dark_count_rate (float, optional): Dark count rate in counts per second
        repetition_rate (float, optional): Source repetition rate in Hz
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): FSO specific parameters dictionary
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    skr_values = []
    
    for mu in mu_values:
        simulator = BBM92Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            channel_mode=channel_mode
        )
        
        # Set FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        skr_per_pulse = simulator.calculate_skr(key_length)
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'g', linewidth=3.5, label=f'SKR at {distance} km')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)', fontsize=25)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=25)
    
    # channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    # plt.title(f'Secret Key Rate vs Mean Photon Number ({channel_type})', fontsize=27)
    
    plt.legend(fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
    return mu_values, skr_values

def plot_qber_vs_distance(distance_values=None, time_window=1e-9, mu=0.1,
                          alice_detector_efficiency=0.20, bob_detector_efficiency=0.20,
                          alice_channel_base_efficiency=1.00, bob_channel_base_efficiency=0.1913,
                          dark_count_rate=1000, channel_mode="fiber", fso_params=None):
    """
    Plot QBER vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        mu (float, optional): Mean photon number
        alice_detector_efficiency (float, optional): Alice's detector efficiency
        bob_detector_efficiency (float, optional): Bob's detector efficiency
        alice_channel_base_efficiency (float, optional): Base efficiency of Alice's channel
        bob_channel_base_efficiency (float, optional): Base efficiency of Bob's channel
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): FSO specific parameters dictionary
    """
    if distance_values is None:
        # For FSO, typically use shorter distances
        if channel_mode == "fso":
            distance_values = np.linspace(0, 20, 20)  # 0-20 km for FSO
        else:
            distance_values = np.linspace(0, 100, 20)  # 0-100 km for fiber
    
    qber_values = []
    
    simulator = BBM92Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        channel_mode=channel_mode
    )
    
    # Set FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'r', linewidth=3.5, label=f'QBER at mu={mu:.2f}')
    plt.axhline(y=5, color='cyan', linestyle='--', label='5% threshold',linewidth=3.5)
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold',linewidth=3.5)
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    
    # channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    # plt.title(f'Quantum Bit Error Rate vs Distance ({channel_type})', fontsize=22)
    
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    
    return distance_values, qber_values

def plot_skr_vs_distance(distance_values=None, time_window=1e-9, key_length=1000000, mu=0.1,
                         alice_detector_efficiency=0.20, bob_detector_efficiency=0.20,
                         alice_channel_base_efficiency=1.00, bob_channel_base_efficiency=0.1913,
                         dark_count_rate=1000, repetition_rate=1000000,
                         channel_mode="fiber", fso_params=None):
    """
    Plot Secret Key Rate vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        key_length (int, optional): Raw key length in bits
        mu (float, optional): Mean photon number
        alice_detector_efficiency (float, optional): Alice's detector efficiency
        bob_detector_efficiency (float, optional): Bob's detector efficiency
        alice_channel_base_efficiency (float, optional): Base efficiency of Alice's channel
        bob_channel_base_efficiency (float, optional): Base efficiency of Bob's channel
        dark_count_rate (float, optional): Dark count rate in counts per second
        repetition_rate (float, optional): Source repetition rate in Hz
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): FSO specific parameters dictionary
    """
    if distance_values is None:
        # For FSO, typically use shorter distances
        if channel_mode == "fso":
            distance_values = np.linspace(0, 20, 20)  # 0-20 km for FSO
        else:
            distance_values = np.linspace(0, 100, 20)  # 0-100 km for fiber
    
    skr_values = []
    
    simulator = BBM92Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        channel_mode=channel_mode
    )
    
    # Set FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
    for distance in distance_values:
        simulator.update_distance(distance)
        skr_per_pulse = simulator.calculate_skr(key_length)
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'm', linewidth=3.5, label=f'SKR at mu={mu:.2f}')
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=20)
    
    # channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    # plt.title(f'Secret Key Rate vs Distance ({channel_type})', fontsize=22)
    
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale('log')  # Log scale often better visualizes SKR decay
    plt.tight_layout()
    plt.show()
    
    return distance_values, skr_values

def compare_fiber_vs_fso(distance_values=None, mu=0.1, time_window=1e-9, key_length=1000000,
                         alice_detector_efficiency=0.20, bob_detector_efficiency=0.20,
                         alice_channel_base_efficiency=1.00, bob_channel_base_efficiency=0.1913,
                         dark_count_rate=1000, repetition_rate=1000000,
                         fso_params=None):
    """
    Compare fiber and FSO performance on the same plot.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        mu (float, optional): Mean photon number
        time_window (float, optional): Detection time window in seconds
        key_length (int, optional): Raw key length in bits
        alice_detector_efficiency (float, optional): Alice's detector efficiency
        bob_detector_efficiency (float, optional): Bob's detector efficiency
        alice_channel_base_efficiency (float, optional): Base efficiency of Alice's channel
        bob_channel_base_efficiency (float, optional): Base efficiency of Bob's channel
        dark_count_rate (float, optional): Dark count rate in counts per second
        repetition_rate (float, optional): Source repetition rate in Hz
        fso_params (dict, optional): FSO specific parameters dictionary
    """
    if distance_values is None:
        distance_values = np.linspace(0, 20, 20)  # 0-20 km is good for comparison
    
    # Initialize simulators
    fiber_simulator = BBM92Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        channel_mode="fiber"
    )
    
    fso_simulator = BBM92Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        channel_mode="fso"
    )
    
    # Set FSO parameters if provided
    if fso_params is not None:
        fso_simulator.set_fso_parameters(**fso_params)
    
    # Calculate QBERs
    fiber_qber_values = []
    fso_qber_values = []
    fiber_skr_values = []
    fso_skr_values = []
    
    for distance in distance_values:
        # Update fiber simulator
        fiber_simulator.update_distance(distance)
        fiber_qber = fiber_simulator.calculate_qber()
        fiber_skr_per_pulse = fiber_simulator.calculate_skr(key_length)
        fiber_skr = fiber_skr_per_pulse * repetition_rate
        
        # Update FSO simulator
        fso_simulator.update_distance(distance)
        fso_qber = fso_simulator.calculate_qber()
        fso_skr_per_pulse = fso_simulator.calculate_skr(key_length)
        fso_skr = fso_skr_per_pulse * repetition_rate
        
        # Store values
        fiber_qber_values.append(fiber_qber)
        fso_qber_values.append(fso_qber)
        fiber_skr_values.append(fiber_skr)
        fso_skr_values.append(fso_skr)
    
    # Plot QBER comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(distance_values, fiber_qber_values, 'b-', linewidth=3.5, label='Fiber')
    plt.plot(distance_values, fso_qber_values, 'r-', linewidth=3.5, label='FSO')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold', linewidth=3.5)
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('QBER (%)', fontsize=14)
    plt.title('QBER Comparison: Fiber vs FSO', fontsize=16)
    plt.legend(fontsize=12)
    
    # Plot SKR comparison
    plt.subplot(1, 2, 2)
    plt.plot(distance_values, fiber_skr_values, 'b-', linewidth=2, label='Fiber')
    plt.plot(distance_values, fso_skr_values, 'r-', linewidth=2, label='FSO')
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('SKR (bits/s)', fontsize=14)
    plt.title('Secret Key Rate Comparison: Fiber vs FSO', fontsize=16)
    plt.legend(fontsize=12)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'distance': distance_values,
        'fiber_qber': fiber_qber_values,
        'fso_qber': fso_qber_values,
        'fiber_skr': fiber_skr_values,
        'fso_skr': fso_skr_values
    }