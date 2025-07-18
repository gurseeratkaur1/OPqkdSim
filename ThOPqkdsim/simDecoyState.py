import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

class DecoyStateQKD:
    def __init__(self, 
                 wavelength=1550,  # nm
                 alpha=0.21,       # dB/km (fiber loss coefficient)
                 e_detector=0.033, # detector error probability (3.3%)
                 Y0=1.7e-6,        # background rate
                 eta_bob=0.045,    # Bob's side efficiency (internal transmittance * detector efficiency)
                 mu=0.5,           # signal state intensity
                 nu1=0.1,          # decoy state 1 intensity
                 nu2=0.0,          # decoy state 2 intensity (vacuum)
                 f=1.1,           # error correction efficiency
                 q=0.5,            # protocol efficiency factor (1/2 for BB84)
                 rep_rate=2e6,     # repetition rate in Hz (2 MHz as default from Table 1)
                 base_efficiency=1.0,   # Base channel efficiency (typically 1.0 for ideal conditions)
                 channel_mode="fiber", # channel mode: "fiber" or "fso"
                 # FSO specific parameters
                 transmitter_diameter=0.1,    # Diameter of transmitter aperture in meters
                 receiver_diameter=0.3,       # Diameter of receiver aperture in meters
                 beam_divergence=0.001,      # Beam divergence angle in radians (1 mrad)
                 atmos_attenuation= 0.2       # Atmospheric attenuation coefficient in dB/km
                ):
        # QKD protocol parameters
        self.wavelength_nm = wavelength
        self.wavelength = wavelength * 1e-9  # Convert to meters for FSO calculations
        self.alpha = alpha
        self.e_detector = e_detector
        self.Y0 = Y0
        self.eta_bob = eta_bob
        self.mu = mu
        self.nu1 = nu1
        self.nu2 = nu2
        self.f = f
        self.q = q
        self.rep_rate = rep_rate
        self.e0 = 0.5  # error rate of background (random)
        
        # Channel parameters
        self.channel_mode = channel_mode
        self.distance = 0  # initialize with zero distance
        # Base efficiency without distance (typically 1.0 for ideal conditions)
        self.base_efficiency = base_efficiency
        
        # FSO specific parameters
        self.transmitter_diameter = transmitter_diameter
        self.receiver_diameter = receiver_diameter
        self.beam_divergence = beam_divergence
        self.atmos_attenuation = atmos_attenuation  # Atmospheric attenuation coefficient in dB/km
        
        # Optical misalignment parameters
        self.misalignment_base = 0.015     # 1.5% base misalignment error
        self.misalignment_factor = 0.0002  # Increase per km
    
    def set_channel_mode(self, mode):
        """
        Set the channel mode to either fiber or FSO
        
        Args:
            mode (str): Either "fiber" or "fso"
        """
        if mode not in ["fiber", "fso"]:
            raise ValueError(f"Unsupported channel mode: {mode}. Use 'fiber' or 'fso'.")
        
        self.channel_mode = mode
    
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
            self.atmos_attenuation = atmos_attenuation

    
    def calculate_fiber_transmittance(self, distance):
        """Calculate channel transmittance for fiber based on distance and loss coefficient"""
        t_AB = self.base_efficiency * (10 ** (-(self.alpha * distance) / 10))
        return t_AB
    
    def calculate_fso_transmittance(self, distance):
        """Calculate channel transmittance for FSO channel"""
        # For zero distance, return direct efficiency without atmospheric effects
        if distance <= 1e-6:  # Effectively zero
            return self.base_efficiency 
    
        # Calculate geometrical loss factor
        beam_diameter_at_receiver = self.transmitter_diameter + (distance * 1000 * self.beam_divergence)
        geo_factor = min(1.0, (self.receiver_diameter / beam_diameter_at_receiver)**2)
        
        #calculate atmospheric loss factor
        atmos_loss = np.exp(-self.atmos_attenuation * distance)

        
        # Calculate overall transmission efficiency
        total_efficiency = (self.base_efficiency * geo_factor * atmos_loss)
        
        return min(1.0, max(0.0, total_efficiency))  # Ensure efficiency is between 0 and 1
    
    def calculate_channel_transmittance(self, distance):
        """Calculate channel transmittance based on channel mode"""
        if self.channel_mode == "fiber":
            return self.calculate_fiber_transmittance(distance)
        elif self.channel_mode == "fso":
            return self.calculate_fso_transmittance(distance)
        else:
            raise ValueError(f"Unsupported channel mode: {self.channel_mode}")
    
    def calculate_total_transmittance(self, distance):
        """Calculate total transmittance including Bob's efficiency"""
        t_AB = self.calculate_channel_transmittance(distance)
        eta = t_AB * self.eta_bob
        return eta
    
    def yield_i_photon(self, i, eta):
        """Calculate yield for i-photon state"""
        # Using Equation (7): Yi = Y0 + ηi - Y0ηi ≈ Y0 + ηi
        # where ηi = 1 - (1 - η)^i from Equation (6)
        eta_i = 1 - (1 - eta) ** i
        Y_i = self.Y0 + eta_i - self.Y0 * eta_i
        return Y_i
    
    def error_i_photon(self, i, eta, distance):
        """Calculate error rate for i-photon state"""
        # Using Equation (9): ei = (e0Y0 + edetectorηi) / Yi
        Y_i = self.yield_i_photon(i, eta)
        eta_i = 1 - (1 - eta) ** i

        e_i = (self.e0 * self.Y0 + self.e_detector * eta_i) / Y_i
        return e_i
    
    def gain_i_photon(self, i, mu, eta):
        """Calculate gain for i-photon state"""
        # Using Equation (8): Qi = Yi * (μ^i/i!) * e^(-μ)
        Y_i = self.yield_i_photon(i, eta)
        Q_i = Y_i * (mu ** i / np.math.factorial(i)) * np.exp(-mu)
        return Q_i
    
    def overall_gain(self, mu, eta):
        """Calculate overall gain"""
        # Using Equation (10): Qμ = Y0 + 1 - e^(-ημ)
        Q_mu = self.Y0 + (1 - np.exp(-eta * mu))
        return Q_mu
    
    def overall_error(self, mu, eta, distance):
        """Calculate overall error """
        # Using Equation (11): EμQμ = e0Y0 + edetector(1 - e^(-ημ))
        Q_mu = self.overall_gain(mu, eta)
        E_mu_Q_mu = self.e0 * self.Y0 + self.e_detector * (1 - np.exp(-eta * mu))
        E_mu = E_mu_Q_mu / Q_mu
        return E_mu
    
    
    def detailed_QBER(self, mu, eta):
        """
        Calculate detailed QBER based on photon number decomposition,
        
        Args:
            mu: Signal state intensity
            eta: Channel transmittance
            
        Returns:
            QBER value (as a ratio, not percentage)
        """
        
        # Calculate probabilities of different photon number states
        p_vacuum = np.exp(-mu)  # Probability of vacuum state
        p_single = mu * np.exp(-mu)  # Probability of single-photon state
        p_multi = 1 - p_vacuum - p_single  # Probability of multi-photon states
        
        # Calculate detection probabilities
        Y_vacuum = self.Y0
        Y_single = eta + self.Y0 - eta * self.Y0
        Y_multi = 1 - (1-eta)**2 + self.Y0 - self.Y0 * (1-(1-eta)**2)
        
        # Calculate gains for each component
        Q_vacuum = p_vacuum * Y_vacuum
        Q_single = p_single * Y_single
        Q_multi = p_multi * Y_multi
        
        # Calculate error rates for each component
        E_vacuum = 0.5  # Random errors for vacuum (dark counts)
        E_single = self.e_detector
        E_multi = self.e_detector * (1 + 0.1 * mu)

        # Calculate overall QBER using weighted average
        total_gain = Q_vacuum + Q_single + Q_multi
        total_error = (Q_vacuum * E_vacuum + Q_single * E_single + Q_multi * E_multi)
        
        qber = total_error / total_gain if total_gain > 0 else 0
        return qber
    
    def estimate_Y0_lower_bound(self, Q_nu1, Q_nu2):
        """Estimate lower bound of Y0 using Equation (18)"""
        Y0_L = max((self.nu1 * Q_nu2 * np.exp(self.nu2) - self.nu2 * Q_nu1 * np.exp(self.nu1)) / 
                   (self.nu1 - self.nu2), 0)
        return Y0_L
    
    def estimate_Y1_lower_bound(self, Q_mu, Q_nu1, Q_nu2, Y0_L):
        """Estimate lower bound of Y1 using Equation (21)"""
        numerator = Q_nu1 * np.exp(self.nu1) - Q_nu2 * np.exp(self.nu2) - \
                    ((self.nu1**2 - self.nu2**2) / self.mu**2) * (Q_mu * np.exp(self.mu) - Y0_L)
        denominator = self.mu * self.nu1 - self.mu * self.nu2 - self.nu1**2 + self.nu2**2
        
        # Handle division by zero
        if abs(denominator) < 1e-10:
            denominator = 1e-10 if denominator >= 0 else -1e-10
            
        Y1_L = (self.mu / denominator) * numerator
        return Y1_L
    
    def estimate_Q1_lower_bound(self, Q_mu, Q_nu1, Q_nu2, Y0_L):
        """Estimate lower bound of Q1 using Equation (22)"""
        Y1_L = self.estimate_Y1_lower_bound(Q_mu, Q_nu1, Q_nu2, Y0_L)
        Q1_L = Y1_L * self.mu * np.exp(-self.mu)
        return Q1_L
    
    def estimate_e1_upper_bound(self, Q_nu1, Q_nu2, E_nu1, E_nu2, Y1_L):
        """Estimate upper bound of e1 using Equation (25)"""
        # Check for division by zero in individual terms
        nu_diff = self.nu1 - self.nu2
        if abs(nu_diff) < 1e-10:
            nu_diff = 1e-10 if nu_diff >= 0 else -1e-10
            
        # Ensure Y1_L is not zero
        if abs(Y1_L) < 1e-10:
            Y1_L = 1e-10
            
        e1_U = (E_nu1 * Q_nu1 * np.exp(self.nu1) - E_nu2 * Q_nu2 * np.exp(self.nu2)) / \
            (nu_diff * Y1_L)
        return e1_U
    
    def key_rate(self, distance):
        """Calculate secure key rate using Equation (26)"""
        eta = self.calculate_total_transmittance(distance)
        
        # Calculate the gains and QBERs for signal and decoy states
        Q_mu = self.overall_gain(self.mu, eta)
        Q_nu1 = self.overall_gain(self.nu1, eta)
        Q_nu2 = self.overall_gain(self.nu2, eta)
        
        # E_mu = self.overall_error(self.mu, eta, distance)
        # E_nu1 = self.overall_error(self.nu1, eta, distance)
        # E_nu2 = self.overall_error(self.nu2, eta, distance)

        E_mu = self.detailed_QBER(self.mu, eta)
        E_nu1 = self.detailed_QBER(self.nu1, eta)
        E_nu2 = self.detailed_QBER(self.nu2, eta)

        # Estimate parameters using decoy state method
        Y0_L = self.estimate_Y0_lower_bound(Q_nu1, Q_nu2)
        Y1_L = self.estimate_Y1_lower_bound(Q_mu, Q_nu1, Q_nu2, Y0_L)
        Q1_L = self.estimate_Q1_lower_bound(Q_mu, Q_nu1, Q_nu2, Y0_L)
        e1_U = self.estimate_e1_upper_bound(Q_nu1, Q_nu2, E_nu1, E_nu2, Y1_L)
        
        # Shannon entropy
        def H2(x):
            """Calculate binary Shannon entropy with safety checks for numerical stability"""
            # Handle values outside [0,1] range by clamping
            if x <= 0 or x >= 1:
                return 0
            
            # Additional protection for values extremely close to 0 or 1
            epsilon = 1e-15  # Small value to prevent log(0)
            x = max(min(x, 1-epsilon), epsilon)
    
            return -x * np.log2(x) - (1-x) * np.log2(1-x)
        
        # Calculate secure key rate using Equation (1)
        R_per_pulse = self.q * (-Q_mu * self.f * H2(E_mu) + Q1_L * (1 - H2(e1_U)))
        R_per_second = R_per_pulse * self.rep_rate
        
        return max(0, R_per_second), E_mu, Q_mu, Y1_L, e1_U, Q1_L


  

def analyze_distance_dependence(qkd, max_distance=150, step=1, channel_mode="fiber"):
    """
    Analyze how key rate and QBER change with distance
    
    Args:
        qkd: DecoyStateQKD object
        max_distance: Maximum distance to analyze in km
        step: Distance step size in km
        channel_mode: "fiber" or "fso"
    """
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    distances = np.arange(0, max_distance + step, step)
    key_rates = []
    qbers = []
    gains = []
    Y1_Ls = []
    e1_Us = []
    Q1_Ls = []
    
    for d in distances:
        rate, qber, gain, Y1_L, e1_U, Q1_L = qkd.key_rate(d)
        key_rates.append(rate)
        qbers.append(qber)
        gains.append(gain)
        Y1_Ls.append(Y1_L)
        e1_Us.append(e1_U)
        Q1_Ls.append(Q1_L)
    
    return distances, key_rates, qbers, gains, Y1_Ls, e1_Us, Q1_Ls

def analyze_mu_dependence(qkd, distance=50, mu_range=(0.1, 1.0), step=0.05, channel_mode="fiber"):
    """
    Analyze how key rate and QBER change with signal state intensity mu
    
    Args:
        qkd: DecoyStateQKD object
        distance: Transmission distance in km
        mu_range: Range of mu values to analyze (min, max)
        step: Step size for mu values
        channel_mode: "fiber" or "fso"
    """
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    mu_values = np.arange(mu_range[0], mu_range[1] + step, step)
    key_rates = []
    qbers = []
    
    original_mu = qkd.mu
    for mu in mu_values:
        qkd.mu = mu
        rate, qber, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
        qbers.append(qber)
    
    qkd.mu = original_mu
    return mu_values, key_rates, qbers

def analyze_decoy_state_intensity(qkd, distance=50, nu1_range=(0.01, 0.3), step=0.02, channel_mode="fiber"):
    """
    Analyze how key rate changes with decoy state intensity nu1
    
    Args:
        qkd: DecoyStateQKD object
        distance: Transmission distance in km
        nu1_range: Range of nu1 values to analyze (min, max)
        step: Step size for nu1 values
        channel_mode: "fiber" or "fso"
    """
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    nu1_values = np.arange(nu1_range[0], nu1_range[1] + step, step)
    key_rates = []
    
    original_nu1 = qkd.nu1
    for nu1 in nu1_values:
        qkd.nu1 = nu1
        rate, _, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
    
    qkd.nu1 = original_nu1
    return nu1_values, key_rates

def analyze_detector_error(qkd, distance=50, error_range=(0.01, 0.1), step=0.005, channel_mode="fiber"):
    """
    Analyze how key rate and QBER change with detector error probability
    
    Args:
        qkd: DecoyStateQKD object
        distance: Transmission distance in km
        error_range: Range of error values to analyze (min, max)
        step: Step size for error values
        channel_mode: "fiber" or "fso"
    """
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    error_values = np.arange(error_range[0], error_range[1] + step, step)
    key_rates = []
    qbers = []
    
    original_error = qkd.e_detector
    for error in error_values:
        qkd.e_detector = error
        rate, qber, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
        qbers.append(qber)
    
    qkd.e_detector = original_error
    return error_values, key_rates, qbers

def plot_key_rate_vs_distance(qkd, max_distance=150, channel_mode="fiber"):
    """
    Plot key rate vs distance
    
    Args:
        qkd: DecoyStateQKD object
        max_distance: Maximum distance to analyze in km
        channel_mode: "fiber" or "fso"
    """
    distances, key_rates, _, _, _, _, _ = analyze_distance_dependence(qkd, max_distance, channel_mode=channel_mode)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(distances, key_rates, 'm', linewidth=3.5,label=f"Secure Key Rate ({channel_mode.upper()})")
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=20)
    #plt.title(f'Secure Key Rate vs Distance - {channel_mode.upper()} Channel', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def plot_qber_vs_distance(qkd, max_distance=150, channel_mode="fiber"):
    """
    Plot QBER vs distance
    
    Args:
        qkd: DecoyStateQKD object
        max_distance: Maximum distance to analyze in km
        channel_mode: "fiber" or "fso"
    """
    distances, _, qbers, _, _, _, _ = analyze_distance_dependence(qkd, max_distance, channel_mode=channel_mode)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, [qber * 100 for qber in qbers], 'r', linewidth=3.5,label=f"QBER ({channel_mode.upper()})")  # Convert to percentage
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    #plt.title(f'QBER vs Distance - {channel_mode.upper()} Channel', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def plot_key_rate_vs_mu(qkd, distance=50, channel_mode="fiber"):
    """
    Plot key rate vs signal state intensity
    
    Args:
        qkd: DecoyStateQKD object
        distance: Transmission distance in km
        channel_mode: "fiber" or "fso"
    """
    mu_values, mu_key_rates, _ = analyze_mu_dependence(qkd, distance=distance, channel_mode=channel_mode)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, mu_key_rates, 'g', linewidth=3.5,label=f"Secure Key Rate ({channel_mode.upper()})")
    plt.xlabel('Signal State Intensity (μ)', fontsize=20)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=20)
    #plt.title(f'Secure Key Rate vs Signal State Intensity at {distance} km - {channel_mode.upper()} Channel', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def plot_key_rate_vs_error(qkd, distance=50, channel_mode="fiber"):
    """
    Plot key rate vs detector error
    
    Args:
        qkd: DecoyStateQKD object
        distance: Transmission distance in km
        channel_mode: "fiber" or "fso"
    """
    error_values, error_key_rates, _ = analyze_detector_error(qkd, distance=distance, channel_mode=channel_mode)
    
    plt.figure(figsize=(10, 6))
    plt.plot(error_values, error_key_rates, 'b', linewidth=3.5,label=f"Secure Key Rate ({channel_mode.upper()})")
    plt.xlabel('Detector Error Probability', fontsize=20)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=20)
    plt.title(f'Secure Key Rate vs Detector Error at {distance} km - {channel_mode.upper()} Channel', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def plot_qber_vs_mu(qkd, distance=50, mu_range=(0.1, 3), step=0.01, channel_mode="fiber"):
    """
    Plot QBER vs signal state intensity (mu) using scientific principles of Decoy State QKD.
    
    Args:
        qkd: DecoyStateQKD object
        distance: Transmission distance in km
        mu_range: Range of mu values to analyze (min, max)
        step: Step size for mu values
        channel_mode: "fiber" or "fso"
    """
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    # Calculate transmittance
    eta = qkd.calculate_total_transmittance(distance)
    
    # Create arrays for plotting
    mu_values = np.arange(mu_range[0], mu_range[1] + step, step)
    qber_values = []
    
    # Use the detailed QBER calculation method from the DecoyStateQKD class
    for mu in mu_values:
        qber = qkd.detailed_QBER(mu, eta)
        qber_values.append(qber * 100)  # Convert to percentage

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'b', linewidth=3.5,label=f"QBER ({channel_mode.upper()})")
    plt.xlabel('Signal State Intensity (μ)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    plt.title(f'QBER vs Signal State Intensity at {distance} km - {channel_mode.upper()} Channel', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


def compare_key_rates(qkd, max_distance=150, step=1):
    """
    Compare key rates between fiber and FSO channels
    
    Args:
        qkd: DecoyStateQKD object
        max_distance: Maximum distance to analyze in km
        step: Distance step size in km
    """
    distances = np.arange(0, max_distance + step, step)
    
    # Get key rates for fiber mode
    qkd.set_channel_mode("fiber")
    fiber_key_rates = []
    for d in distances:
        rate, _, _, _, _, _ = qkd.key_rate(d)
        fiber_key_rates.append(rate)
    
    # Get key rates for FSO mode
    qkd.set_channel_mode("fso")
    fso_key_rates = []
    for d in distances:
        rate, _, _, _, _, _ = qkd.key_rate(d)
        fso_key_rates.append(rate)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.semilogy(distances, fiber_key_rates, 'b-', linewidth=2.5, label="Fiber Channel")
    plt.semilogy(distances, fso_key_rates, 'r--', linewidth=2.5, label="FSO Channel")
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=20)
    plt.title('Comparison of Secure Key Rates: Fiber vs FSO', fontsize=22)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

def compare_qbers(qkd, max_distance=150, step=1):
    """
    Compare QBERs between fiber and FSO channels
    
    Args:
        qkd: DecoyStateQKD object
        max_distance: Maximum distance to analyze in km
        step: Distance step size in km
    """
    distances = np.arange(0, max_distance + step, step)
    
    # Get QBERs for fiber mode
    qkd.set_channel_mode("fiber")
    fiber_qbers = []
    for d in distances:
        _, qber, _, _, _, _ = qkd.key_rate(d)
        fiber_qbers.append(qber * 100)  # Convert to percentage
    
    # Get QBERs for FSO mode
    qkd.set_channel_mode("fso")
    fso_qbers = []
    for d in distances:
        _, qber, _, _, _, _ = qkd.key_rate(d)
        fso_qbers.append(qber * 100)  # Convert to percentage
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(distances, fiber_qbers, 'b-', linewidth=2.5, label="Fiber Channel")
    plt.plot(distances, fso_qbers, 'r--', linewidth=2.5, label="FSO Channel")
    plt.xlabel('Distance (km)', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    plt.title('Comparison of QBERs: Fiber vs FSO', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

def analyze_fso_parameters(qkd, distance=50, parameter_name="beam_divergence", 
                          value_range=None, step=None, unit=""):
    """
    Analyze how an FSO parameter affects key rate
    
    Args:
        qkd: DecoyStateQKD object
        distance: Transmission distance in km
        parameter_name: Name of the FSO parameter to analyze
        value_range: Range of values to analyze (min, max)
        step: Step size for values
        unit: Unit of the parameter for display purposes
    """
    # Set default ranges for different parameters
    if value_range is None or step is None:
        if parameter_name == "beam_divergence":
            value_range = (0.0001, 0.005)  # 0.1 to 5 mrad
            step = 0.0001
            unit = "rad"
        elif parameter_name == "transmitter_diameter":
            value_range = (0.01, 0.3)  # 1 to 30 cm
            step = 0.01
            unit = "m"
        elif parameter_name == "receiver_diameter":
            value_range = (0.05, 0.5)  # 5 to 50 cm
            step = 0.02
            unit = "m"
        elif parameter_name == "pointing_error":
            value_range = (1e-7, 1e-5)  # 0.1 to 10 μrad
            step = 5e-7
            unit = "rad"
        else:
            raise ValueError(f"Unsupported parameter: {parameter_name}")
    
    # Ensure we're in FSO mode
    qkd.set_channel_mode("fso")
    
    # Create parameter range
    param_values = np.arange(value_range[0], value_range[1] + step, step)
    key_rates = []
    qbers = []
    
    # Store original parameter value
    original_value = getattr(qkd, parameter_name)
    
    # Analyze each parameter value
    for value in param_values:
        # Set the parameter value
        setattr(qkd, parameter_name, value)
        
        # Calculate key rate and QBER
        rate, qber, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
        qbers.append(qber * 100)  # Convert to percentage
    
    # Restore original value
    setattr(qkd, parameter_name, original_value)
    
    # Create key rate plot
    plt.figure(figsize=(12, 7))
    plt.plot(param_values, key_rates, 'go-', linewidth=2)
    plt.xlabel(f'{parameter_name.replace("_", " ").title()} ({unit})', fontsize=20)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=20)
    plt.title(f'Key Rate vs {parameter_name.replace("_", " ").title()} at {distance} km', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    # Create QBER plot
    plt.figure(figsize=(12, 7))
    plt.plot(param_values, qbers, 'ro-', linewidth=2)
    plt.xlabel(f'{parameter_name.replace("_", " ").title()} ({unit})', fontsize=20)
    plt.ylabel('QBER (%)', fontsize=20)
    plt.title(f'QBER vs {parameter_name.replace("_", " ").title()} at {distance} km', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()