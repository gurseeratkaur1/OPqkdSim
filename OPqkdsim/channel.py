import numpy as np
from scipy.fft import fft, ifft
from .timeline import Timeline
import random

class QuantumChannel(Timeline):
    """
    Quantum Optical Fiber Channel with real-time event handling.

    Models signal propagation considering:
    - Attenuation
    - Chromatic dispersion (second-order and third-order)
    - Differential group delay
    - Kerr non-linearity (Self-Phase Modulation)
    - Split-Step Fourier Method (SSFM) for simulation

    Args:
        fiber_length (float): Length of the fiber (km).
        attenuation (float): Attenuation per km (dB/km).
        beta2 (float): Second-order dispersion (ps²/km).
        beta3 (float): Third-order dispersion (ps³/km).
        dg_delay (float): Differential group delay (ps/km).
        gamma (float): Non-linear coefficient (W⁻¹·km⁻¹).
        fft_samples (int): Number of FFT samples.
        step_size (float): Step size for SSFM (km).
        timeline (Timeline): Event-driven simulation framework.
    """

    def __init__(self, 
                timeline, 
                fiber_length= 50.0, 
                attenuation= 0.2, 
                beta2=-21.27, 
                beta3= 0.12, 
                dg_delay= 0.1, 
                gamma= 1.3, 
                fft_samples= 1000, # remember to match this with steps in solve_dynamics in DFB laser
                refractive_index=1.5,
                step_size= 0.1 ):
        super().__init__()
        self.timeline = timeline
        self.fiber_length = fiber_length
        self.attenuation = 10 ** (-attenuation / 10)  # Convert dB/km to linear scale
        self.beta2 = beta2 * 1e-24  # Convert ps²/km to s²/m
        self.beta3 = beta3 * 1e-36  # Convert ps³/km to s³/m
        self.dg_delay = dg_delay * 1e-12  # Convert ps/km to s/m
        self.gamma = gamma
        self.fft_samples = fft_samples
        self.step_size = step_size
        self.refractive_index = refractive_index  # refractive index

        # Frequency domain representation
        self.frequency_grid = np.fft.fftfreq(fft_samples, d=1e-12)  # Assume 1 ps sampling interval
        self.angular_freq = 2 * np.pi * self.frequency_grid  # Convert to angular frequency

    def dispersion_operator(self, dz):
        """
        Computes the dispersion operator in the frequency domain for a step dz.

        Args:
            dz (float): Propagation step size in km.

        Returns:
            np.ndarray: Dispersion transfer function.
        """
        w = self.angular_freq
        D = np.exp(
            -1j * (self.beta2 / 2) * (w ** 2) * dz - 1j * (self.beta3 / 6) * (w ** 3) * dz
        )
        return D

    def nonlinear_operator(self, E, dz):
        """
        Computes the non-linear phase shift (Self-Phase Modulation) for step dz.

        Args:
            E (np.ndarray): Optical field.
            dz (float): Step size (km).

        Returns:
            np.ndarray: Modified optical field.
        """
        return E * np.exp(1j * self.gamma * np.abs(E) ** 2 * dz)

    def compute_propagation_delay(self):
        """Computes time delay for photon propagation through fiber."""
        c = 3e8  # Speed of light in vacuum (m/s)
        L = self.fiber_length * 1e3  # Convert km to meters
        delay = (L * self.refractive_index) / c  # Time in seconds
        return delay

    def propagate_signal(self, event_time, t, P, Ex, Ey, E):
        """
        Simulates optical signal propagation through the fiber using SSFM.

        Args:
            event_time (float): Time event was triggered.
            t (np.ndarray): Time array.
            P (np.ndarray): Optical power.
            Ex (np.ndarray): X-polarized field.
            Ey (np.ndarray): Y-polarized field.
            E (np.ndarray): Total field.
        """
        num_steps = int(self.fiber_length / self.step_size)
        dz = self.step_size

        # Convert power to amplitude
        E_complex = Ex + 1j * Ey
        #print(E_complex)
        for _ in range(num_steps):
            # Apply half nonlinearity
            E_complex = self.nonlinear_operator(E_complex, dz / 2)

            # Apply dispersion in frequency domain
            E_complex_f = fft(E_complex)
            E_complex_f *= self.dispersion_operator(dz)
            E_complex = ifft(E_complex_f)

            # Apply second half of nonlinearity
            E_complex = self.nonlinear_operator(E_complex, dz / 2)

            # Apply attenuation
            E_complex *= self.attenuation ** dz

        # Extract real and imaginary parts
        Ex_out, Ey_out = E_complex.real, E_complex.imag
        P_out = Ex_out ** 2 + Ey_out ** 2  # Updated Power

        print(f"[{event_time:.3e} s] Signal propagated: P_out={P_out[-1]:.2e} W")

        # Compute propagation delay
        propagation_delay = self.compute_propagation_delay()
        #print(propagation_delay)
        # Schedule event for signal arrival after delay
        arrival_time = event_time + propagation_delay

        self.timeline.publish_delay(propagation_delay,self,None, P_out, Ex_out, Ey_out, np.sqrt(Ex_out**2 + Ey_out**2))
        
        # self.timeline.schedule_event(
        #     propagation_delay, self.timeline.publish, self, P_out, Ex_out, Ey_out, np.sqrt(Ex_out**2 + Ey_out**2)
        # )

class ClassicalChannel(Timeline):
    """
    Simulates a classical communication channel for BB84.
    
    Models:
    - Transmission delay
    - Packet loss
    - Noise corruption
    - Secure communication for basis reconciliation and error correction

    Args:
        timeline (Timeline): Event-driven simulation framework.
        transmission_speed (float): Speed of classical communication (km/ms).
        loss_rate (float): Probability of packet loss.
        noise_prob (float): Probability of bit flips due to noise.
        encryption (bool): Whether messages are encrypted for security.
    """

    def __init__(self, timeline, reciever, transmission_speed=200, loss_rate=0.01, noise_prob=0.001, encryption=False):
        super().__init__()
        self.timeline = timeline
        self.reciever = reciever
        self.transmission_speed = transmission_speed  # Classical signals travel near speed of light
        self.loss_rate = loss_rate  # Packet loss probability
        self.noise_prob = noise_prob  # Bit flip probability
        self.encryption = encryption  # Secure message exchange
        
    def compute_transmission_delay(self, distance):
        """
        Computes the delay based on the communication distance.

        Args:
            distance (float): Distance in km.
        
        Returns:
            float: Delay in milliseconds.
        """
        return distance / self.transmission_speed
    
    def transmit_message(self, message, distance=0.0025):
        """
        Simulates message transmission with loss, delay, and noise.

        Args:
            sender (str): Name of the sender (Alice/Bob).
            receiver (str): Name of the receiver (Bob/Alice).
            message (dict): Data being transmitted.
            distance (float): Distance between sender and receiver.
        """
        delay = self.compute_transmission_delay(distance)

        # Simulate packet loss
        if random.random() < self.loss_rate:
            print(f"[{self.timeline.get_current_time():.3f} ms] Packet lost during transmission.")
            return

        # Simulate noise (bit flips in message)
        noisy_message = self.apply_noise(message)

        # Simulate encryption (optional)
        if self.encryption:
            secure_message = self.encrypt_message(noisy_message)
        else:
            secure_message = noisy_message

        # Schedule message delivery event
        self.timeline.publish_delay(delay, self, self.reciever, secure_message)
    
    def apply_noise(self, message):
        """
        Simulates bit flips due to classical noise.

        Args:
            message (dict): The message containing classical bits.

        Returns:
            dict: Noisy message with potential bit flips.
        """
        noisy_message = message.copy()
        for key, value in noisy_message.items():
            if isinstance(value, np.ndarray):  # Check if the value is an array of bits
                flip_mask = np.random.rand(len(value)) < self.noise_prob
                noisy_message[key] = np.bitwise_xor(value, flip_mask)  # Apply bit flips
        return noisy_message
    
    def encrypt_message(self, message):
        """
        Simulates encryption for secure message transfer.

        Args:
            message (dict): Message dictionary.

        Returns:
            dict: Encrypted message.
        """
        # Simple XOR-based encryption with a dummy key (for simulation)
        encryption_key = np.random.randint(0, 2, len(message["bits"]), dtype=np.uint8)
        encrypted_bits = np.bitwise_xor(message["bits"], encryption_key)
        return {"bits": encrypted_bits, "key": encryption_key}
    
    def decrypt_message(self, encrypted_message):
        """
        Simulates decryption of the message.

        Args:
            encrypted_message (dict): Encrypted message.

        Returns:
            dict: Decrypted message.
        """
        decrypted_bits = np.bitwise_xor(encrypted_message["bits"], encrypted_message["key"])
        return {"bits": decrypted_bits}
    

