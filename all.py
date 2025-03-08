import time
import heapq
import numpy as np
from scipy.fft import fft, ifft
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import scipy.special as sps
from scipy.linalg import hadamard
from scipy.stats import poisson, chi2_contingency, norm
import os
import secrets

class Timeline:
    """
    Real-time Timeline class that manages and executes scheduled events.
    It ensures simulation timing aligns with real-world execution time.
    """

    def __init__(self, dt=1e-3):
        """
        Initializes the timeline.
        
        Args:
            dt (float): Default time step for calculations (seconds).
        """
        self.t_start = time.time()  # Capture real-world start time
        self.current_time = self.t_start  # Initialize current time
        self.dt = dt
        self.event_queue = []  # Min-heap (priority queue) for events
        self.subscribers = {}  # Store component event handlers

    def schedule_event(self, delay, event_function, *args, **kwargs):
        """
        Schedules an event with a real-time delay.

        Args:
            delay (float): Delay in seconds from the current time.
            event_function (callable): Function to execute.
            *args, **kwargs: Additional parameters for the function.
        """
        event_time = time.time() + delay        
        heapq.heappush(self.event_queue, (event_time, event_function, args, kwargs))

    def execute_events(self, max_events=np.inf):
        """
        Executes scheduled events in real-time.

        Args:
            max_events (int, optional): Maximum number of events to process. 
                                        If None, runs until the queue is empty.
        """
        event_count = 0

        while self.event_queue:
            if max_events and event_count >= max_events:
                break

            event_time, event_function, args, kwargs = heapq.heappop(self.event_queue)
            time_to_wait = max(0, event_time - time.time())  # Avoid negative delay

            if time_to_wait > 0:
                time.sleep(time_to_wait)  # Real-time delay

            self.current_time = time.time()  # Update current time
            event_function(self.current_time, *args, **kwargs)  # Execute event
            event_count += 1

    def get_current_time(self):
        """Returns the current real-world simulation time."""
        return time.time()

    def get_elapsed_time(self):
        """Returns elapsed simulation time since start."""
        return time.time() - self.t_start

    def target_publish(self, sender, target=None, *args, **kwargs):
        """Forwards signals selectively to the correct component.
        
        Args:
            sender: The component sending the signal.
            target: The specific component to receive the signal (if provided).
            *args, **kwargs: Additional data to forward.
        """
        if sender in self.subscribers:
            if target:
                # Send to a specific target if it exists
                if target in self.subscribers[sender]:  
                    self.schedule_event(self.dt, target, *args, **kwargs)
                else:
                    print(f"Warning: Target {target} is not subscribed to {sender}. Ignoring.")
            else:
                # Default to broadcasting if no target is specified
                for handler in self.subscribers[sender]:
                    self.schedule_event(self.dt, handler, *args, **kwargs)
            
    def publish(self, sender, *args, **kwargs):
        """Forwards signals to all subscribed components of `sender`."""
        if sender in self.subscribers:
            for handler in self.subscribers[sender]:  # ideally only one handler, if multiple use target_publish
                self.schedule_event(self.dt, handler, *args, **kwargs)

    def publish_delay(self,delay,sender,target, *args, **kwargs):
        """Forwards signals selectively to the correct component.
        
        Args:
            delay: The amount of time to delay
            sender: The component sending the signal.
            target: The specific component to receive the signal (if provided).
            *args, **kwargs: Additional data to forward.
        """
        if sender in self.subscribers:
            if target:
                # Send to a specific target if it exists
                if target in self.subscribers[sender]:  
                    self.schedule_event(delay, target, *args, **kwargs)
                else:
                    print(f"Warning: Target {target} is not subscribed to {sender}. Ignoring.")
            else:
                # Default to broadcasting if no target is specified
                for handler in self.subscribers[sender]:
                    self.schedule_event(delay, handler, *args, **kwargs)

    def subscribe(self, component, handler):
        """Registers a component to receive signals from a sender."""
        if component not in self.subscribers:
            self.subscribers[component] = []  # Initialize empty list if first time
        self.subscribers[component].append(handler)  # Append 


class CRNG:
    def __init__(self, method="hardware"):
        """
        Initialize QRNG with a chosen method:
        - "hardware": Uses os.urandom() for true randomness.
        - "crypto": Uses secrets.randbits() for cryptographic randomness.
        """
        self.method = method

    def random_bit(self):
        """Generates a single random bit (0 or 1) using the chosen method."""
        if self.method == "hardware":
            return ord(os.urandom(1)) % 2  # Uses hardware-based entropy
        elif self.method == "crypto":
            return secrets.randbits(1)  # Uses cryptographic randomness
        else:
            raise ValueError("Invalid QRNG method. Choose 'hardware' or 'crypto'.")

    def generate_bits(self, num_bits=1):
        """Generates a list of random bits of specified length."""
        return [self.random_bit() for _ in range(num_bits)]

    def generate_basis(self, num_bases=1):
        """Generates a list of random basis choices (0: Rectilinear, 1: Diagonal)."""
        return [self.random_bit() for _ in range(num_bases)]


class LightSource(Timeline):
    """
    Generic Light Source class that inherits from Timeline.
    Handles common laser properties and scheduling.
    """

    def __init__(self, timeline, wavelength=1550e-9, power=0, pulse_rate=1e9, dt=1e-12):
        """
        Initializes the light source.

        Args:
            wavelength (float): Laser wavelength (m).
            power (float): Optical power output (W).
            pulse_rate (float): Repetition rate of pulses (Hz).
            dt (float): Time step for simulation (s).
        """
        super().__init__(dt)
        self.timeline = timeline
        self.wavelength = wavelength
        self.power = power
        self.pulse_rate = pulse_rate  # How often pulses are emitted

    def emit_light(self):
        """Generic method for emitting light (to be implemented by subclasses)."""
        raise NotImplementedError("emit_light() must be implemented in subclasses")



class DFBLaser(LightSource):
    """
    Distributed Feedback (DFB) Laser model, inherits LightSource.
    """

    def __init__(
        self,
        timeline,
        I_t=lambda t: 5e-3,  # Default: 5 mA constant injection current
        V_a=1e-16,  # Active region volume (m^3)
        tau_n=2e-9,  # Carrier lifetime (s)
        tau_p=1e-12,  # Photon lifetime (s)
        g0=1e-5,  # Gain coefficient (m^3/s)
        N0=1e24,  # Transparency carrier density (m^-3)
        epsilon_C=1e-3,  # Gain compression factor
        beta=1e-4,  # Spontaneous emission factor
        Gamma=0.5,  # Optical confinement factor
        eta_DFB=0.95,  # Differential quantum efficiency
        hnu=1.5e-19,  # Photon energy (J)
        lambda_0=1550e-9,  # Default wavelength (m)
        pulse_rate=1e9,  # Default pulse rate (Hz)
        mu=0.1,  # Mean photon number per pulse
        dt=1e-12,
        shots=100 # Number of times to emit pulses of light before stopping
    ):
        super().__init__(timeline, wavelength=lambda_0, power=0, pulse_rate=pulse_rate, dt=dt)
        self.I_t = I_t
        self.V_a = V_a
        self.tau_n = tau_n
        self.tau_p = tau_p
        self.g0 = g0
        self.N0 = N0
        self.epsilon_C = epsilon_C
        self.beta = beta
        self.Gamma = Gamma
        self.lambda_0 = lambda_0
        self.eta_DFB = eta_DFB
        self.hnu = hnu
        self.mu = mu
        self.shots = shots

    def rate_equations(self, t, y):
        """Defines the rate equations for carrier and photon densities.
        Args:
            t (float): Time
            y (array): [Carrier density N, Photon density S]

        Returns:
            [dN/dt, dS/dt] (array): Time derivatives 
        """
        N, S = y
        I = self.I_t(t)  # Injection current at time t
        
        dN_dt = (I / (1.6e-19 * self.V_a)) - (N / self.tau_n) - (
            self.g0 * (N - self.N0) / (1 + self.epsilon_C * S) * S
        )
        
        dS_dt = (self.Gamma * self.g0 * (N - self.N0) / (1 + self.epsilon_C * S) - 1 / self.tau_p) * S + (
            self.beta * self.Gamma * N / self.tau_n
        )
        
        if dS_dt < 0:
            dS_dt = max(0, dS_dt)  # Prevent negative photon production


        return [dN_dt, dS_dt]

    def solve_dynamics(self, duration=1e-9, steps=1000):
        """Computes optical power and electric field components with true randomness."""
        t_eval = np.linspace(0, duration, steps)
        sol = solve_ivp(self.rate_equations, (0, duration), [self.N0, 1e10], t_eval=t_eval)
        
        t = sol.t
        N, S = sol.y

        # Randomize the final photon number using Poisson sampling
        S_final = 0 
        while S_final == 0 :
            S_final = poisson.rvs(mu=self.mu)  # Ensure at least 1 photon # Sample photon number from Poisson distribution

        # Scale photon density based on random photon count
        S = (S / S[-1]) * S_final  # Normalize and rescale S

        # Compute Optical Power: P = η_DFB * S * hν / τ_p
        P = self.eta_DFB * S * self.hnu / self.tau_p

        # Compute Electric Field Components
        Ex = np.sqrt(P) * np.cos(2 * np.pi * t / self.lambda_0)
        Ey = np.sqrt(P) * np.sin(2 * np.pi * t / self.lambda_0)
        E = np.sqrt(Ex**2 + Ey**2)  # Total Electric Field

        return t, P, Ex, Ey, E

    def emit_light(self, event_time):
        """Computes the laser's optical pulse and propagates it."""

        if self.shots == 0:
            return  # Stop emitting if no more shots are left
        
        t, P, Ex, Ey, E = self.solve_dynamics()
        #t, P, Ex, Ey, E = t, P[-1], Ex[-1], Ey[-1], E[-1]
        #print(P)
        print(f"[{event_time:.5e} s] Pulse emitted at {self.lambda_0} m, Power: {P[-1]:.5e} W, Shots left:{self.shots}")

        # Propagate to next component (e.g., Optical Fiber)
        self.timeline.publish(self, P, Ex, Ey, E)

        # Schedule next pulse
        self.shots -= 1
        if self.shots > 0:
            next_pulse_time = 1 / self.pulse_rate
            self.timeline.schedule_event(next_pulse_time, self.emit_light)

    def initialize_laser(self):
        """Starts the laser emission schedule."""
        pulse_time = 1 / self.pulse_rate
        self.timeline.schedule_event(pulse_time, self.emit_light)

    
    # def analyze_randomness(self, output_list):
    #     """Analyzes whether 0s and 1s are generated randomly."""
    #     num_zeros = output_list.count(0)
    #     num_ones = output_list.count(1)
    #     total_samples = len(output_list)

    #     print("\n--- Randomness Analysis ---")
    #     print(f"Total Samples: {total_samples}")
    #     print(f"0s: {num_zeros} ({num_zeros/total_samples:.2%})")
    #     print(f"1s: {num_ones} ({num_ones/total_samples:.2%})")

    #     # Expected counts for a uniform distribution
    #     expected = [total_samples / 2, total_samples / 2]
    #     observed = [num_zeros, num_ones]

    #     # Chi-square test for uniformity
    #     chi2_stat, p_value = chi2_contingency([observed, expected])[:2]

    #     print(f"Chi-Square Statistic: {chi2_stat:.2f}")
    #     print(f"P-value: {p_value:.4f}")

    #     if p_value > 0.05:
    #         print("✅ The generated bits are likely random.")
    #     else:
    #         print("⚠️ The generated bits may not be truly random.")



class PolarizationController(Timeline):
    """
    Polarization Controller (PC) for adjusting the polarization state of a pulse.
    
    Ensures that the amplitudes along the crystal axes of the phase modulator 
    are equal by transforming the input electric field.
    """

    def __init__(self, timeline, theta=0, compensate=False):
        """
        Initializes the polarization controller.

        Args:
            timeline (Timeline): Simulation timeline for event management.
            theta (float): Rotation angle of the waveplate (in radians).
            compensate (Boolean): compensate for birefringence-induced SOP drifts in the quantum channel
        """
        self.timeline = timeline
        self.theta = theta  # Default waveplate rotation angle
        self.compensate = compensate

    def jones_matrix1(self):
        """Returns the Jones matrix for the polarization transformation."""
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        return np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

    def jones_matrix2(self):
        """
        Returns the Jones matrix for compensating SOP drift and birefringence effects.

        The transformation ensures that polarization drifts along the LiNO₃ crystal axes 
        are swapped, effectively reversing their effect.
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # Apply a 90-degree phase shift to swap the LiNO₃ axes
        compensation_matrix = np.array([
            [cos_theta, sin_theta],
            [sin_theta, -cos_theta]
        ])

        return compensation_matrix

    def process_signal(self, event_time, P, Ex, Ey, E):
        """
        Adjusts polarization to ensure equal amplitudes along modulator axes.

        Args:
            event_time (float): Event timestamp.
            P (float): Optical power.
            Ex (float): Electric field component along x.
            Ey (float): Electric field component along y.
            E (float): Total electric field magnitude.
        """
        # Convert input to Jones vector
        E_in = np.array([Ex, Ey])

        # Apply transformation
        if self.compensate:
            E_out = np.dot(self.jones_matrix2(), E_in)
        else:
            E_out = np.dot(self.jones_matrix1(), E_in)
        #print(E_out.shape)
        # Normalize power
        E_out /= np.linalg.norm(E_out) / np.sqrt(P)
        # Avoid division by zero or NaN
        # norm_E_out = np.linalg.norm(E_out)
        # sqrt_P = np.sqrt(P)
        # if norm_E_out > 0 and sqrt_P > 0:
        #     E_out /= norm_E_out / sqrt_P
        # else:
        #     print(f"Warning: Invalid division encountered. norm_E_out={norm_E_out}, sqrt_P={sqrt_P}")
        #     E_out = np.array([0, 0])  # Default safe value to prevent NaN
        

        # Publish the transformed polarization state to the next component
        self.timeline.publish(self, P, E_out[0], E_out[1], np.linalg.norm(E_out))

        print(f"[{event_time:.5e} s] Polarization Adjusted: Ex={E_out[0,-1]:.5e}, Ey={E_out[1,-1]:.5e}")



class AlicePolarizationModulator:
    """
    Phase Modulator for BB84.
    - Applies appropriate phase shift.
    - Uses externally generated basis
    - Uses externally provided bits
    """

    def __init__(self, timeline, V_pi=3.5, bias=0,modulation_list=None, bit_list=None):
        self.timeline = timeline
        self.V_pi = V_pi  # Half-wave voltage for phase shift π
        self.bias = bias  # Bias phase offset
        #self.random_generator = CRNG()

        # Store modulation sequence in an external list
        if modulation_list is None:
            raise ValueError("A modulation list must be provided for basis values.")
        self.modulation_list = modulation_list
        if bit_list is None:
            raise ValueError("A bit list must be provided for encoding phases.")
        self.modulation_list = modulation_list # Use externally generated basis
        self.bit_list = bit_list  # Use externally generated bits
        self.index = 0  # Keep track of which bit is being used


    def extract_bit(self):
        """Extracts the next bit and basis from the provided list."""
        if self.index >= len(self.bit_list) :
            raise IndexError("Not enough bits in the list.")
        if self.index >= len(self.modulation_list):
            raise IndexError("Not enough basis in the list.")

        bit = self.bit_list[self.index]
        basis = self.modulation_list[self.index]
        self.index += 1
        return basis, bit

    def phase_shift(self, basis, bit):
        """Calculates the phase shift based on basis and bit."""
        if basis == 0:
            return np.pi * bit + self.bias  # 0 or π
        elif basis == 1:
            return (np.pi / 2) * (1 - 2 * bit) + self.bias  # π/2 or -π/2

    def process_signal(self, event_time, P, Ex, Ey, E):
        if len(self.bit_list) == 0:
            raise ValueError("Bit list cannot be empty.")
        if len(self.modulation_list) == 0:
            raise ValueError("Modulation list cannot be empty.")
        
        basis,bit = self.extract_bit()  # Get the next bit from the lightsource  list
        phi = self.phase_shift(basis, bit)

        Ex_mod = Ex * np.exp(1j * phi)
        Ey_mod = Ey * np.exp(1j * phi)

        self.timeline.publish(self, event_time, P, Ex_mod.real, Ey_mod.real, np.abs(E))
        print(f"[{event_time:.5e} s] Phase Modulated: Basis={basis}, Bit={bit}, φ={phi:.5f} rad")

class BobPolarizationModulator:
    """
    Polarization Modulator at Bob's end.
    - Applies phase shift to rotate polarization states accordingly.
    - Passes the modified polarization state to the next component (PBS).
    - Stores basis in a user-provided list.
    """

    def __init__(self, timeline, V_pi=3.5, bias=0, modulation_list=None):
        """
        Initializes the polarization modulator.

        Args:
            timeline (Timeline): The event-driven simulation timeline.
            V_pi (float): Half-wave voltage for phase shift π.
            bias (float): Optional phase offset.
            modulation_list (list): list to store basis values
        """
        self.timeline = timeline
        self.V_pi = V_pi
        self.bias = bias
        self.index = 0
        # Store modulation sequence in an external list
        if modulation_list is None:
            raise ValueError("A modulation list must be provided to access basis values.")
        self.modulation_list = modulation_list

    def extract_basis(self):
        """Extracts the next basis from the provided list."""
        
        if self.index >= len(self.modulation_list):
            raise IndexError("Not enough basis in the list.")

        basis = self.modulation_list[self.index]
        self.index += 1
        return basis
    
    def phase_shift(self, basis):
        """Applies the appropriate phase shift to align measurement axes."""
        if basis == 0:
            return 0  # No shift needed for Z-basis measurement
        elif basis == 1:
            return np.pi / 4  # Rotate by 45° to align with X-basis

    def process_signal(self, event_time, P, Ex, Ey, E):
        """
        Processes incoming quantum states by selecting a random measurement basis
        and applying a phase shift to rotate polarization.

        Args:
            event_time (float): Simulation timestamp.
            P (float): Optical power.
            Ex (float): Electric field component along x.
            Ey (float): Electric field component along y.
            E (float): Total electric field magnitude.
        """
        if len(self.modulation_list) == 0:
            raise ValueError("Modulation list cannot be empty.")
        
        basis = self.extract_basis()
        phi = self.phase_shift(basis)

        # Apply phase shift to align measurement basis
        Ex_mod = Ex * np.exp(1j * phi)
        Ey_mod = Ey * np.exp(1j * phi)

        # Forward the adjusted polarization state to the next component (PBS)
        self.timeline.publish(self, P, Ex_mod.real, Ey_mod.real, np.abs(E))

        print(f"[{event_time:.5e} s] Basis: {basis}, Applied Phase Shift: {phi:.5f} rad")

class VariableOpticalAttenuator:
    """
    Variable Optical Attenuator (VOA) class that dynamically controls optical power.
    
    The VOA applies a variable attenuation factor to the incoming optical pulse
    and then propagates the modified signal.
    """

    def __init__(self, timeline, attenuation_dB=0):
        """
        Initializes the VOA.

        Args:
            timeline (Timeline): The timeline instance managing events.
            attenuation_dB (float): Initial attenuation level in decibels (dB).
        """
        self.timeline = timeline
        self.attenuation_dB = attenuation_dB  # Initial attenuation level

    def process_signal(self, event_time, t, P, Ex, Ey, E):
        """
        Applies attenuation to the incoming optical pulse and propagates the modified signal.

        Args:
            event_time (float): Time at which the event is processed.
            t (array): Time samples of the pulse.
            P (array): Optical power of the pulse.
            Ex (array): X-polarized component of the electric field.
            Ey (array): Y-polarized component of the electric field.
            E (array): Total electric field amplitude.
        """
        # Convert attenuation from dB to linear scale
        attenuation_factor = 10 ** (-self.attenuation_dB / 10)

        # Apply attenuation
        P_att = P * attenuation_factor
        Ex_att = Ex * np.sqrt(attenuation_factor)
        Ey_att = Ey * np.sqrt(attenuation_factor)
        E_att = E * np.sqrt(attenuation_factor)
        #print(P_att.shape)
        print(f"[{event_time:.5e} s] VOA applied {self.attenuation_dB:.5f} dB attenuation. New Power: {P_att[-1]:.5e} W")

        # Propagate to next component
        self.timeline.publish(self,t, P_att, Ex_att, Ey_att, E_att)    


class PolarizationBeamSplitter(Timeline):
    """
    Polarization Beam Splitter (PBS) for Bob in BB84.
    
    - Separates horizontal and vertical polarization components.
    - Passes horizontal (H) polarization straight.
    - Reflects vertical (V) polarization to a different path.
    - Uses Bob's basis choice (Z or X) to determine the correct detectors.
    - If Bob chose Z-basis → Uses H/V Detectors.
    - If Bob chose X-basis → Uses D+/D- Detectors.
    """

    def __init__(self, timeline, detector_HV, detector_D, bob_modulation_list=None):
        """
        Args:
            timeline (Timeline): Simulation timeline.
            detector_H (Detector): Detector for horizontal polarization (Z-basis).
            detector_V (Detector): Detector for vertical polarization (Z-basis).
            detector_D_plus (Detector): Detector for +45° polarization (X-basis).
            detector_D_minus (Detector): Detector for -45° polarization (X-basis).
            bob_modulation_list (list): User-provided list storing Bob's basis choices.
        """
        super().__init__()
        self.timeline = timeline
        self.detector_HV = detector_HV
        # self.detector_V = detector_V
        self.detector_D = detector_D
        # self.detector_D_minus = detector_D_minus
        self.detection_index = 0  # Tracks which basis to use
        if bob_modulation_list is None:
            raise ValueError("A modulation list must be provided to access basis values.")
        self.bob_modulation_list = bob_modulation_list

    def process_signal(self, event_time, P, Ex, Ey, E):
        """
        Processes incoming photon based on Bob's chosen basis.
        
        Args:
            event_time (float): Timestamp of the signal.
            P (float): Total optical power.
            Ex (float): Electric field component along x (H polarization).
            Ey (float): Electric field component along y (V polarization).
            E (float): Total electric field magnitude.
        """
        if self.detection_index >= len(self.bob_modulation_list):
            print(f"[{event_time:.3e} s] PBS: No more basis choices left, ignoring photon.")
            return

        bob_basis = self.bob_modulation_list[self.detection_index]  # Get Bob's basis choice
        self.detection_index += 1  # Move to the next photon

        # Compute horizontal (H) and vertical (V) power components
        P_H = Ex**2  # Horizontal polarization
        P_V = Ey**2  # Vertical polarization

        # Normalize total power
        # P_H = P_H * (P / (P_H + P_V)) if (P_H + P_V) != 0 else 0
        # P_V = P_V * (P / (P_H + P_V)) if (P_H + P_V) != 0 else 0
        P_H = np.where((P_H + P_V) != 0, P_H * (P / (P_H + P_V)), 0)
        P_V = np.where((P_H + P_V) != 0, P_V * (P / (P_H + P_V)), 0)


        if bob_basis == 0:
            # If Bob chose the Z-basis, send photon to H/V detectors
            print(f"[{event_time:.3e} s] PBS Output - Basis: {bob_basis}, P_H: {P_H[-1]:.2e} W, P_V: {P_V[-1]:.2e} W")  

            self.timeline.target_publish(self, self.detector_HV.detect_photon, P_H, Ex, 0, np.abs(Ex))
            #self.timeline.target_publish(self, self.detector_V.detect_photon, P_V, 0, Ey, np.abs(Ey))
        elif bob_basis == 1:
            # Compute diagonal basis components
            P_D_plus = (P_H + P_V) / 2 + (Ex * Ey)  # +45° polarization
            P_D_minus = (P_H + P_V) / 2 - (Ex * Ey)  # -45° polarization

            # Normalize power
            # P_D_plus = P_D_plus * (P / (P_D_plus + P_D_minus)) if (P_D_plus + P_D_minus) != 0 else 0
            # P_D_minus = P_D_minus * (P / (P_D_plus + P_D_minus)) if (P_D_plus + P_D_minus) != 0 else 0
            P_D_plus = np.where((P_D_plus + P_D_minus) != 0, P_D_plus * (P / (P_D_plus + P_D_minus)), 0)
            P_D_minus = np.where((P_D_plus + P_D_minus) != 0, P_D_minus * (P / (P_D_plus + P_D_minus)), 0)

            print(f"[{event_time:.3e} s] PBS Output - P_D+: {P_D_plus[-1]:.2e} W, P_D-: {P_D_minus[-1]:.2e} W")
            
            self.timeline.target_publish(self, self.detector_D.detect_photon,P_D_plus, Ex, Ey, np.abs(E))
            #self.timeline.target_publish(self, self.detector_D_minus.detect_photon,P_D_minus, Ex, Ey, np.abs(E))



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
                attenuation= 0.05, 
                beta2=-17, 
                beta3= 0.12, 
                dg_delay= 0.1, 
                gamma= 0.8, 
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


class SinglePhotonDetector:
    """
    Simulates a Single Photon Detector (SPD)
    
    Models detection probability using Poisson distribution, dark counts, after-pulsing, and dead time.
    """

    def __init__(self, 
                 timeline, 
                 name="SPD", 
                 qe=0.95, 
                 dark_count_rate=1000, 
                 jitter_mean=30e-12, 
                 jitter_std=5e-12, 
                 dead_time=25e-9, 
                 alice_bits = None,
                 alice_basis = None,
                 bob_basis = None,
                 bit_list=None):
        """
        Initializes the SPD.

        Args:
            timeline (Timeline): Simulation timeline for scheduling events.
            name (str): Detector name.
            qe (float): Quantum efficiency (0 to 1).
            dark_count_rate (float): Dark count rate (Hz).
            jitter_mean (float): Mean detector jitter (seconds).
            jitter_std (float): Standard deviation of jitter (seconds).
            dead_time (float): Detector dead time after a detection (seconds).
            bit_list (list): User-provided list to store detected events.
        """
        self.timeline = timeline
        self.name = name
        self.qe = qe
        self.dark_count_rate = dark_count_rate
        self.jitter_mean = jitter_mean
        self.jitter_std = jitter_std
        self.dead_time = dead_time
        self.bit_list = bit_list if bit_list is not None else []

        self.last_detection_time = -np.inf  # Last detection event time
        self.p0 = 0.0317  # After-pulsing initial probability
        self.a = 0.00115  # After-pulsing decay parameter
        self.after_pulsing_prob = self.p0  # Initialize after-pulsing probability

        if alice_bits is None or alice_basis is None or bob_basis is None:
            raise ValueError("A bits and basis list must be provided to access values.")

        self.alice_bits = alice_bits
        self.alice_basis = alice_basis
        self.bob_basis = bob_basis
        self.index = 0
        # Schedule to schedule dark count generation separately when instantiating 
        
    def extract_basis(self):
        """Extracts the next basis from the provided list."""
        
        if self.index >= len(self.alice_basis):
            raise IndexError("Not enough basis in the list.")

        basis_alice = self.alice_basis[self.index]
        basis_bob = self.bob_basis[self.index]
        self.index += 1
        return basis_alice == basis_bob
    
    def detect_photon(self, event_time, power, Ex, Ey, E, background_photons=0):
        """
        Handles the detection of an incoming photon using a probabilistic model.

        Args:
            event_time (float): Timestamp of the photon arrival.
            power : Optical power of the photon.
            Ex : Electric field component along x-axis.
            Ey : Electric field component along y-axis.
            E : Total electric field magnitude.
            background_photons : Additional background photon noise.
        """

        # If inputs are arrays, extract last value (latest photon event)
        if isinstance(power, np.ndarray):
            power = power[-1]
        if isinstance(E, np.ndarray): E = E[-1]
        if isinstance(Ex, np.ndarray):
            Ex = Ex[-1]
        if isinstance(Ey, np.ndarray):
            Ey = Ey[-1]

        # Check dead time
        if event_time - self.last_detection_time < self.dead_time:
            print(f"[{event_time:.9f} s] {self.name}: Detector in dead time. Photon ignored.")
            self.bit_list.append(0)  # No detection during dead time
            return

        # Compute Mean Photon Number (MPN)
        MPN = self.qe * power  # ηµ
        total_photons = MPN + background_photons

        # Probability of at least one photon in pulse (Poisson model)
        Pp = 1 - np.exp(-total_photons)

        # Dark count probability (Poisson model)
        dark_counts = np.random.poisson(self.dark_count_rate * self.timeline.dt)
        Pd = 1 - np.exp(-dark_counts)  # Probability of at least one dark count event
        #Pd = 0
        # Compute Pclick using Eq. (21) with proper probabilities
        Pclick = (Pp + self.after_pulsing_prob + Pd 
                  - Pp * self.after_pulsing_prob 
                  - self.after_pulsing_prob * Pd 
                  - Pd * Pp 
                  + Pp * self.after_pulsing_prob * Pd)
        #rand = np.random.rand()
        rand = 0
        # Probabilistic detection decision
        if  rand < Pclick:
            # Introduce jitter (Gaussian delay)
            detection_delay = np.random.normal(self.jitter_mean, self.jitter_std)
            detection_time = event_time + detection_delay

            print(f"[{detection_time:.9f} s] {self.name}: Photon detected! Pclick {Pclick} random {rand}")

            # Register detection and update after-pulsing probability
            #self.timeline.schedule_event(detection_delay, self.register_detection, detection_time)
            self.last_detection_time = detection_time
            self.after_pulsing_prob *= np.exp(-self.a)  # Decay after-pulsing probability

            if self.extract_basis():
                # Basis matches: Bob gets Alice's bit with 100% accuracy
                self.bit_list.append(self.alice_bits[self.index - 1])
            else:
                detected_bit = np.random.choice([0, 1])
                self.bit_list.append(detected_bit)  # Store detected bit
        else:
            print(f"[{event_time:.9f} s] {self.name}: Photon missed. Pclick {Pclick} random {rand}")
            self.bit_list.append(0)  # Store missed detection as 0



class SiftedKey(Timeline):
    """
    Computes the sifted key for Alice after receiving Bob's bases.
    
    Args:
        timeline (Timeline): Event-driven simulation framework.
        alice_bits (np.ndarray): Alice's original bits.
        alice_bases (np.ndarray): Alice's chosen bases.
        sifted_key (list or np.ndarray): Mutable array to store the sifted key.
    """
    def __init__(self, timeline, alice_bits, alice_bases, sifted_key=None,matching_indices = None):
        self.timeline = timeline
        self.alice_bits = np.array(alice_bits)  # Ensure NumPy array
        self.alice_bases = np.array(alice_bases)

        # Validate user-provided storage
        if sifted_key is None:
            raise ValueError("A sifted_key list or NumPy array must be provided to store the key.")
        if matching_indices is None:
            raise ValueError("A matching indices list or NumPy array must be provided to store.")
        self.sifted_key = sifted_key  # Reference to external storage
        self.matching_indices = matching_indices

    def receive_bob_bases(self, event_time, message):
        """
        Receives Bob's bases from the classical channel and computes the sifted key.
        """
        if isinstance(message, dict):
            bob_bases_received = np.array(message["basis"])  # Ensure NumPy array
        else:
            bob_bases_received = np.array(message)
        
        # Identify matching bases
        matching_indices = np.where(bob_bases_received == self.alice_bases)[0]
        
        # Compute sifted key
        sifted_key_values = self.alice_bits[matching_indices]

        # Store in user-provided array (supports both lists and NumPy arrays)
        if isinstance(self.sifted_key, list):
            self.sifted_key.clear()  # Clear existing values
            self.sifted_key.extend(sifted_key_values.tolist())  # Append new values
        elif isinstance(self.sifted_key, np.ndarray):
            self.sifted_key[:] = sifted_key_values  # Modify NumPy array in place
        else:
            raise TypeError("sifted_key must be a mutable list or NumPy array.")
        
        if isinstance(self.matching_indices, list):
            self.matching_indices.clear()  # Clear existing values
            self.matching_indices.extend(matching_indices.tolist())  # Append new values
        elif isinstance(self.matching_indices, np.ndarray):
            self.matching_indices[:] = matching_indices  # Modify NumPy array in place
        else:
            raise TypeError("matching_indices must be a mutable list or NumPy array.")

        print(f"Alice computed sifted key of length {len(self.sifted_key)}.")

        # Publish sifted key for next steps (e.g., LDPC error reconciliation)
        # self.timeline.publish(self, self.sifted_key)



class LDPCReconciliation:
    """
    Implements LDPC Error Reconciliation for Bob's sifted key.
    
    Args:
        sifted_key (list): Bob's sifted key as a binary list.
        code_rate (float): LDPC code rate (e.g., 0.8 means 80% information, 20% redundancy).
    """
    def __init__(self, sifted_key, code_rate=0.8):
        self.sifted_key = np.array(sifted_key)
        self.n = len(self.sifted_key)  # Total code length
        self.k = int(self.n * code_rate)  # Number of information bits
        self.H, self.G = self.generate_ldpc_matrices(self.n, self.k)  # Generate parity-check and generator matrices
    
    def generate_ldpc_matrices(self, n, k):
        """Generates a simple LDPC-like parity-check matrix H and generator matrix G."""
        np.random.seed(42)  # Ensure reproducibility
        
        # Create a random sparse parity-check matrix H
        H = np.random.randint(0, 2, size=(n-k, n))  # (n-k) parity equations for n bits
        
        # Create a generator matrix G (must satisfy H * G^T = 0)
        G = np.eye(k, n, dtype=int)  # Start with an identity matrix for message bits
        for i in range(n-k):  
            G = np.vstack((G, H[i]))  # Append parity check rows
        
        return H, G

    def compute_parity_bits(self):
        """Encodes Bob's sifted key using LDPC and extracts parity bits."""
        codeword = np.dot(self.G, self.sifted_key) % 2  # Generate codeword
        parity_bits = codeword[self.k:]  # Extract parity bits (redundant bits)
        return parity_bits

    def correct_errors(self, alice_sifted_key):
        """
        Alice corrects errors using iterative decoding (simple parity-check method).
        
        Args:
            alice_sifted_key (list): Alice's sifted key before error correction.
            received_parity (list): Parity bits received from Bob over the classical channel.
        
        Returns:
            np.ndarray: Corrected sifted key.
        """
        alice_sifted_key = np.array(alice_sifted_key)
        max_iter = 10  # Maximum number of decoding iterations

        for _ in range(max_iter):
            syndrome = np.dot(self.H, alice_sifted_key) % 2  # Compute syndrome
            if np.all(syndrome == 0):  
                break  # No errors detected
            
            error_indices = np.where(syndrome == 1)[0]
            alice_sifted_key[error_indices] ^= 1  # Flip bits to correct errors

        return alice_sifted_key

    def publish_parity(self, parity_bits):
        """Simulates sending the parity bits through the classical channel."""
        print(f"Bob sent parity bits: {parity_bits.tolist()}")


class UhashPA:
    """
    Implements Privacy Amplification using Universal Hashing (Toeplitz matrices).
    """

    def __init__(self, output_length=256):
        self.output_length = output_length

    def generate_toeplitz_matrix(self, input_length):
        """
        Generates a Toeplitz matrix for universal hashing.
        """
        rand_values = np.random.randint(0, 2, input_length + self.output_length - 1)
        toeplitz_matrix = sp.diags(
            [rand_values[i:input_length + i] for i in range(self.output_length)], offsets=np.arange(self.output_length)
        ).toarray()
        return toeplitz_matrix % 2  # Binary matrix

    def apply_hashing(self, key):
        """
        Applies Toeplitz matrix hashing for privacy amplification.
        """
        input_length = len(key)
        toeplitz_matrix = self.generate_toeplitz_matrix(input_length)
        return (toeplitz_matrix @ key) % 2

    def amplify(self, reconciled_key):
        """
        Reduces key length while improving secrecy.
        """
        return self.apply_hashing(reconciled_key)


class QBERCalculator:
    """
    Class to calculate the Quantum Bit Error Rate (QBER) and statistical error.
    """

    def __init__(self, sifted_key_length, sample_bits, confidence=0.99):
        self.sifted_key_length = sifted_key_length  # Total bits in sifted key
        self.sample_bits = sample_bits  # Number of sample bits to estimate QBER
        self.confidence = confidence  # Confidence level (S)
        self.qber = None
        self.statistical_error = None
        self.approximated_qber = None

    def calculate_qber(self, alice_key, bob_key):
        """
        Computes the QBER by comparing a random sample of Alice and Bob's keys.
        """
        sample_indices = np.random.choice(self.sifted_key_length, self.sample_bits, replace=False)
        sample_errors = np.sum(alice_key[sample_indices] != bob_key[sample_indices])

        self.qber = sample_errors / self.sample_bits
        return self.qber

    def calculate_statistical_error(self):
        """
        Computes the statistical error in QBER estimation using Equation 2.4.
        """
        #self.statistical_error = np.sqrt(self.qber * (1 - self.qber) / (self.sample_bits * (1 - self.confidence)))
        self.statistical_error = np.sqrt(self.qber * (1 - self.qber) / self.sample_bits)

        return self.statistical_error

    def calculate_approximated_qber(self):
        """
        Computes the approximated QBER using Equation 2.5.
        """
        if self.qber is None or self.statistical_error is None:
            raise ValueError("QBER and statistical error must be calculated first.")
        
        self.approximated_qber = self.qber + self.statistical_error
        print(f"approximated_qber = {self.qber} ± {self.statistical_error}")
        return self.approximated_qber


class SecretKeyRateCalculator:
    """
    Class to calculate the Secret Key Rate (SKR) based on the amount of information revealed to Eve.
    """

    def __init__(self, sifted_key_length, total_time_slots, error_correction_bits, eve_allowed_fraction=0.01):
        self.sifted_key_length = sifted_key_length  # LBits
        self.total_time_slots = total_time_slots  # N
        self.error_correction_bits = error_correction_bits  # LParity
        self.eve_allowed_fraction = eve_allowed_fraction  # EAllowed
        self.final_key_length = None
        self.secret_key_rate = None

    def calculate_bits_revealed_to_eve(self, qber, dark_count_probability, pns_attack_bits):
        """
        Computes the total number of bits revealed to Eve.
        """
        # Intercept-resend attack (LIR)
        intercept_resend_bits = np.ceil(2 * (qber - dark_count_probability) * self.sifted_key_length)

        # Total bits revealed to Eve (LEve)
        total_eve_bits = intercept_resend_bits + pns_attack_bits + self.error_correction_bits
        return total_eve_bits

    def calculate_final_key_length(self, total_eve_bits):
        """
        Computes the final key length after privacy amplification.
        """
        privacy_amplification_bits = np.ceil((total_eve_bits - self.eve_allowed_fraction * self.sifted_key_length) / 
                                             (1 - self.eve_allowed_fraction))
        
        self.final_key_length = self.sifted_key_length - privacy_amplification_bits
        return self.final_key_length

    def calculate_secret_key_rate(self):
        """
        Computes the secret key rate (SKR).
        """
        if self.final_key_length is None:
            raise ValueError("Final key length must be calculated first.")

        self.secret_key_rate = self.final_key_length / self.total_time_slots
        return self.secret_key_rate