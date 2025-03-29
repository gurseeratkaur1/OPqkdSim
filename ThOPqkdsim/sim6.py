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
    Represents the quantum channel between source and receiver.
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


class BB84Simulator:
    """
    Simulates the BB84 QKD protocol with type-II SPDC source.
    """
    def __init__(self, mu, alice_detector_efficiency, bob_detector_efficiency, 
                 alice_channel_base_efficiency, bob_channel_base_efficiency,
                 dark_count_rate, time_window, distance=0, attenuation=0.2, p_eve=0.0):
        """
        Initialize the BB84 simulator.
        
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
        """
        self.source = PhotonSource(mu)
        self.mu = mu
        self.alice_channel = Channel(alice_channel_base_efficiency, 0, attenuation)  # Assuming source is at Alice
        self.bob_channel = Channel(bob_channel_base_efficiency, distance, attenuation)
        self.alice_detector = Detector(alice_detector_efficiency, dark_count_rate, time_window)
        self.bob_detector = Detector(bob_detector_efficiency, dark_count_rate, time_window)
        self.distance = distance
        self.attenuation = attenuation
        self.p_eve = p_eve
        self.n_max = 10  # Maximum photon number to consider in calculations
    
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
    
    def calculate_joint_probabilities(self):
        """
        Calculate joint probabilities P(l,α,β) for all relevant photon numbers.
        
        Returns:
            dict: Dictionary with (l,α,β) tuples as keys and probabilities as values
        """
        joint_probs = {}
        ps = self.source.photon_distribution(self.n_max)
        
        for l in range(self.n_max + 1):
            for alpha in range(l + 1):
                for beta in range(l + 1):
                    p_alice = self.alice_channel.transmission_probability(l, alpha)
                    p_bob = self.bob_channel.transmission_probability(l, beta)
                    joint_probs[(l, alpha, beta)] = p_alice * p_bob * ps[l]
        
        return joint_probs
    
    def calculate_p_rec(self):
        """
        Calculate the probability that both Alice and Bob's detectors record a click.
        
        Returns:
            float: Probability of recording a coincidence
        """
        joint_probs = self.calculate_joint_probabilities()
        p_an4 = 1 - (1 - self.alice_detector.p_noise)**4
        p_bn4 = 1 - (1 - self.bob_detector.p_noise)**4
        
        # P(l > 0, α > 0, β > 0)
        p_both_photons = sum(prob for (l, alpha, beta), prob in joint_probs.items() 
                            if l > 0 and alpha > 0 and beta > 0)
        
        # P(l > 0, α = 0, β > 0)
        p_alice_noise_bob_photon = sum(prob for (l, alpha, beta), prob in joint_probs.items() 
                                     if l > 0 and alpha == 0 and beta > 0)
        
        # P(l > 0, α > 0, β = 0)
        p_alice_photon_bob_noise = sum(prob for (l, alpha, beta), prob in joint_probs.items() 
                                     if l > 0 and alpha > 0 and beta == 0)
        
        # Final calculation
        p_rec = (p_both_photons + 
                p_alice_noise_bob_photon * p_an4 + 
                p_alice_photon_bob_noise * p_bn4 + 
                p_an4 * p_bn4)
        
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
        p_an4 = 1 - (1 - self.alice_detector.p_noise)**4
        p_bn4 = 1 - (1 - self.bob_detector.p_noise)**4
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
        # Formula based on BB84 information theory (simplified)
        if qber >= 0.11:  # No secure key possible if QBER too high
            return 0
        
        # Calculate secret key fraction using simplified asymptotic formula
        r = max(0, 1 - 2 * self.h_binary(qber))
        
        # Calculate final secret key rate
        skr = p_key * r  #* key_length # for bits per channel use
        
        return skr
    
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


def plot_qber_vs_mu(mu_values=None, time_window=1e-9, distance=10,
                    alice_detector_efficiency = 0.20,bob_detector_efficiency = 0.20,
                    alice_channel_base_efficiency = 1.00, bob_channel_base_efficiency = 0.1913, 
                    dark_count_rate = 1000 ):
    """
    Plot QBER vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float, optional): Distance in kilometers
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    qber_values = []
    
    # Based on the provided parameters from the document
    # alice_detector_efficiency = 0.20
    # bob_detector_efficiency = 0.20
    # alice_channel_base_efficiency = 1.00  # Source is at Alice
    # bob_channel_base_efficiency = 0.1913
    # dark_count_rate = 1000  # 1000 counts per second
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance
        )
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Mean Photon Number ')
    plt.legend()
    plt.savefig('qber_vs_mu.png')
    plt.show()
    
    return mu_values, qber_values


def plot_skr_vs_mu(mu_values=None, time_window=1e-9, key_length=1000000, distance=10,
                alice_detector_efficiency = 0.20,
                bob_detector_efficiency = 0.20,
                alice_channel_base_efficiency = 1.00, 
                bob_channel_base_efficiency = 0.1913,
                dark_count_rate = 1000 ):
    """
    Plot Secret Key Rate vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        key_length (int, optional): Raw key length in bits
        distance (float, optional): Distance in kilometers
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 20)
    
    skr_values = []
    
    # Based on the provided parameters from the document
    # alice_detector_efficiency = 0.20
    # bob_detector_efficiency = 0.20
    # alice_channel_base_efficiency = 1.00  # Source is at Alice
    # bob_channel_base_efficiency = 0.1913
    # dark_count_rate = 1000  # 1000 counts per second
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance
        )
        skr = simulator.calculate_skr(key_length)
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('Secret Key Rate (bits per channel use)')
    plt.title(f'Secret Key Rate vs Mean Photon Number')
    plt.savefig('skr_vs_mu.png')
    plt.show()
    
    return mu_values, skr_values


def plot_qber_vs_distance(distance_values=None, time_window=1e-9, mu=0.1,
                        alice_detector_efficiency = 0.20,
                        bob_detector_efficiency = 0.20,
                        alice_channel_base_efficiency = 1.00,
                        bob_channel_base_efficiency = 0.1913,
                        dark_count_rate = 1000 ):
    """
    Plot QBER vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        mu (float, optional): Mean photon number
    """
    if distance_values is None:
        distance_values = np.linspace(0, 100, 20)
    
    qber_values = []
    
    # Based on the provided parameters from the document
    # alice_detector_efficiency = 0.20
    # bob_detector_efficiency = 0.20
    # alice_channel_base_efficiency = 1.00  # Source is at Alice
    # bob_channel_base_efficiency = 0.1913
    # dark_count_rate = 1000  # 1000 counts per second
    
    simulator = BB84Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0  # Will be updated in the loop
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'ro-', linewidth=2)
    plt.grid(True)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Distance')
    plt.legend()
    plt.savefig('qber_vs_distance.png')
    plt.show()
    
    return distance_values, qber_values


def plot_skr_vs_distance(distance_values=None, time_window=1e-9, key_length=1000000, mu=0.1,
                        alice_detector_efficiency = 0.20,
                        bob_detector_efficiency = 0.20,
                        alice_channel_base_efficiency = 1.00,
                        bob_channel_base_efficiency = 0.1913,
                        dark_count_rate = 1000 ):
    """
    Plot Secret Key Rate vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        key_length (int, optional): Raw key length in bits
        mu (float, optional): Mean photon number
    """
    if distance_values is None:
        distance_values = np.linspace(0, 100, 20)
    
    skr_values = []
    
    # Based on the provided parameters from the document
    # alice_detector_efficiency = 0.20
    # bob_detector_efficiency = 0.20
    # alice_channel_base_efficiency = 1.00  # Source is at Alice
    # bob_channel_base_efficiency = 0.1913
    # dark_count_rate = 1000  # 1000 counts per second
    
    simulator = BB84Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0  # Will be updated in the loop
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        skr = simulator.calculate_skr(key_length)
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    #plt.semilogy(distance_values, skr_values, 'mo-', linewidth=2)
    plt.plot(distance_values, skr_values, 'mo-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secret Key Rate (bits per channel use)')
    plt.title('Secret Key Rate vs Distance ')
    plt.savefig('skr_vs_distance.png')
    plt.show()
    
    return distance_values, skr_values

