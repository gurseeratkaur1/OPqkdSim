import time
import heapq
import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft

class Timeline:
    """
    Real-time Timeline class that manages and executes scheduled events.
    It ensures simulation timing aligns with real-world execution time.
    """

    def __init__(self, dt=1e-12):
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

    def execute_events(self, max_events=None):
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

    def publish(self, sender, *args, **kwargs):
        """Forwards signals to the next subscribed component."""
        if sender in self.subscribers:
            handler = self.subscribers[sender]
            self.schedule_event(self.dt, handler, *args, **kwargs)

    def subscribe(self, component, handler):
        """Registers a component to receive signals."""
        self.subscribers[component] = handler

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
        V_a=1e-16,  # Active region volume (m³)
        τ_n=2e-9,  # Carrier lifetime (s)
        τ_p=1e-12,  # Photon lifetime (s)
        g0=1e-5,  # Gain coefficient (m³/s)
        N0=1e24,  # Transparency carrier density (m⁻³)
        ε_C=1e-3,  # Gain compression factor
        β=1e-4,  # Spontaneous emission factor
        Γ=0.5,  # Optical confinement factor
        η_DFB=0.8,  # Differential quantum efficiency
        hν=1.5e-19,  # Photon energy (J)
        λ0=1550e-9,  # Default wavelength (m)
        pulse_rate=1e9,  # Default pulse rate (Hz)
        dt=1e-12
    ):
        super().__init__(timeline,wavelength=λ0, power=0, pulse_rate=pulse_rate, dt=dt)
        self.I_t = I_t
        self.V_a = V_a
        self.τ_n = τ_n
        self.τ_p = τ_p
        self.g0 = g0
        self.N0 = N0
        self.ε_C = ε_C
        self.β = β
        self.Γ = Γ
        self.η_DFB = η_DFB
        self.hν = hν

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
        
        dN_dt = (I / (1.6e-19 * self.V_a)) - (N / self.τ_n) - (
            self.g0 * (N - self.N0) / (1 + self.ε_C * S) * S
        )
        
        dS_dt = (self.Γ * self.g0 * (N - self.N0) / (1 + self.ε_C * S) - 1 / self.τ_p) * S + (
            self.β * self.Γ * N / self.τ_n
        )
        
        return [dN_dt, dS_dt]

    def solve_dynamics(self, duration=1e-9, steps=1000):
        """Computes optical power and electric field components."""
        t_eval = np.linspace(0, duration, steps)
        sol = solve_ivp(self.rate_equations, (0, duration), [self.N0, 1e10], t_eval=t_eval)
        
        t = sol.t
        N, S = sol.y

        # Compute Optical Power: P = η_DFB * S * hν / τ_p
        P = self.η_DFB * S * self.hν / self.τ_p

        # Compute Electric Field Components
        Ex = np.sqrt(P) * np.cos(2 * np.pi * t / self.λ0)
        Ey = np.sqrt(P) * np.sin(2 * np.pi * t / self.λ0)
        E = np.sqrt(Ex**2 + Ey**2)  # Total Electric Field

        return t, P, Ex, Ey, E

    def emit_light(self, event_time):
        """Computes the laser's optical pulse and propagates it."""
        t, P, Ex, Ey, E = self.solve_dynamics()
        print(f"[{event_time:.3e} s] Pulse emitted at {self.λ0} m, Power: {P[-1]:.2e} W")

        # Propagate to next component (e.g., Optical Fiber)
        self.timeline.publish(t, P, Ex, Ey, E)

class InLinePolariser(Timeline):
    """Filters one polarization mode and applies insertion loss."""

    def __init__(self, timeline, dt=1e-12, ILPloss=0.5, PER=30):
        super().__init__(dt)
        self.timeline = timeline
        self.ILP_factor = 10 ** (-ILPloss / 10)  
        self.PER_factor = 10 ** (-PER / 10)  

    def process_signal(self, event_time, t, P, Ex, Ey, E):
        """Applies polarization filtering and publishes the signal for the next component."""
        Ex_out = Ex * np.sqrt(self.ILP_factor)  
        Ey_out = Ey * np.sqrt(self.PER_factor)  
        P_out = (Ex_out**2 + Ey_out**2)  

        print(f"[{event_time:.3e} s] Polariser Output - Power: {P_out[-1]:.2e} W (PER applied: {self.PER_factor:.2e})")

        # Publish the processed signal to all outputs
        self.timeline.publish(t, P_out, Ex_out, Ey_out, np.sqrt(Ex_out**2 + Ey_out**2))


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

        # Normalize power
        E_out /= np.linalg.norm(E_out) / np.sqrt(P)

        # Publish the transformed polarization state to the next component
        self.timeline.publish(self, event_time, P, E_out[0], E_out[1], np.linalg.norm(E_out))

        print(f"[{event_time:.3e} s] Polarization Adjusted: Ex={E_out[0]:.3e}, Ey={E_out[1]:.3e}")

class RandomBitGenerator:
    """Generates random bits for BB84 encoding."""

    def generate_bit(self):
        return np.random.choice([0, 1])

    def generate_basis(self):
        """Chooses a random basis: 0 (Z-basis) or 1 (X-basis)."""
        return np.random.choice(["Z", "X"])


class AlicePolarizationModulator:
    """
    Lithium Niobate (LiNO₃) Phase Modulator for BB84.
    - Randomly selects a basis (X or Z).
    - Generates a random bit (0 or 1).
    - Applies appropriate phase shift.
    - Stores basis and bit in a user-provided list.
    """

    def __init__(self, timeline, V_pi=3.5, bias=0, modulation_list=None):
        self.timeline = timeline
        self.V_pi = V_pi  # Half-wave voltage for phase shift π
        self.bias = bias  # Bias phase offset
        self.random_generator = RandomBitGenerator()

        # Store modulation sequence in an external list
        if modulation_list is None:
            raise ValueError("A modulation list must be provided to store basis and bit values.")
        self.modulation_list = modulation_list

    def generate_random_modulation(self):
        """Generates a random basis and bit, then stores them in the user-provided list."""
        basis = self.random_generator.generate_basis()
        bit = self.random_generator.generate_bit()
        self.modulation_list.append((basis, bit))
        return basis, bit

    def phase_shift(self, basis, bit):
        """Calculates the phase shift based on basis and bit."""
        if basis == "Z":
            return np.pi * bit + self.bias  # 0 or π
        elif basis == "X":
            return (np.pi / 2) * (1 - 2 * bit) + self.bias  # π/2 or -π/2

    def process_signal(self, event_time, P, Ex, Ey, E):
        basis, bit = self.generate_random_modulation()
        phi = self.phase_shift(basis, bit)

        Ex_mod = Ex * np.exp(1j * phi)
        Ey_mod = Ey * np.exp(1j * phi)

        self.timeline.publish(self, event_time, P, Ex_mod.real, Ey_mod.real, np.abs(E))
        print(f"[{event_time:.3e} s] Phase Modulated: Basis={basis}, Bit={bit}, φ={phi:.3f} rad")

class BobPolarizationModulator:
    """
    Polarization Modulator at Bob's end.
    - Randomly selects a measurement basis (Z or X).
    - Applies phase shift to rotate polarization states accordingly.
    - Passes the modified polarization state to the next component (PBS).
    """

    def __init__(self, timeline, V_pi=3.5, bias=0):
        """
        Initializes the polarization modulator.

        Args:
            timeline (Timeline): The event-driven simulation timeline.
            V_pi (float): Half-wave voltage for phase shift π.
            bias (float): Optional phase offset.
        """
        self.timeline = timeline
        self.V_pi = V_pi
        self.bias = bias
        self.random_generator = RandomBitGenerator()

    def generate_random_basis(self):
        """Chooses a random measurement basis: Z (0°/90°) or X (45°/-45°)."""
        return self.random_generator.generate_basis()

    def phase_shift(self, basis):
        """Applies the appropriate phase shift to align measurement axes."""
        if basis == "Z":
            return 0  # No shift needed for Z-basis measurement
        elif basis == "X":
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
        basis = self.generate_random_basis()
        phi = self.phase_shift(basis)

        # Apply phase shift to align measurement basis
        Ex_mod = Ex * np.exp(1j * phi)
        Ey_mod = Ey * np.exp(1j * phi)

        # Forward the adjusted polarization state to the next component (PBS)
        self.timeline.publish(self, event_time, P, Ex_mod.real, Ey_mod.real, np.abs(E))

        print(f"[{event_time:.3e} s] Basis Chosen: {basis}, Applied Phase Shift: {phi:.3f} rad")

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

        print(f"[{event_time:.3e} s] VOA applied {self.attenuation_dB:.2f} dB attenuation. New Power: {P_att[-1]:.2e} W")

        # Propagate to next component
        self.timeline.publish(self, t, P_att, Ex_att, Ey_att, E_att)    


class PolarizationBeamSplitter(Timeline):
    """
    Polarization Beam Splitter (PBS) for Bob.
    
    - Separates horizontal and vertical polarization components.
    - Passes horizontal (H) polarization straight.
    - Reflects vertical (V) polarization to a different path.
    - Sends signals to the appropriate detectors (or further processing units).
    """

    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline  # Simulation timeline

    def process_signal(self, event_time, P, Ex, Ey, E):
        """
        Splits incoming light into horizontal and vertical components.
        
        Args:
            event_time (float): Timestamp of the signal.
            P (float): Total optical power.
            Ex (float): Electric field component along x (H polarization).
            Ey (float): Electric field component along y (V polarization).
            E (float): Total electric field magnitude.
        """
        # Compute horizontal (H) and vertical (V) power components
        P_H = Ex**2  # Horizontal component passes straight
        P_V = Ey**2  # Vertical component is reflected

        # Normalize total power
        P_H = P_H * (P / (P_H + P_V)) if (P_H + P_V) != 0 else 0
        P_V = P_V * (P / (P_H + P_V)) if (P_H + P_V) != 0 else 0

        print(f"[{event_time:.3e} s] PBS Output - P_H: {P_H:.2e} W, P_V: {P_V:.2e} W")

        # Forward H and V polarization components to their respective detectors
        self.timeline.publish("Horizontal_Detector", event_time, P_H, Ex, 0, np.abs(Ex))
        self.timeline.publish("Vertical_Detector", event_time, P_V, 0, Ey, np.abs(Ey))

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
                fft_samples= 1024, 
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

        # Publish processed signal to next component
        self.timeline.publish(self, event_time, P_out, Ex_out, Ey_out, np.sqrt(Ex_out**2 + Ey_out**2))


class SinglePhotonDetector(Timeline):
    """
    Simulates a Single Photon Detector (SPD) that detects incoming photons 
    with efficiency, jitter, dark counts, and dead time.

    Integrates with the Timeline class using the publisher-subscriber model.
    """

    def __init__(self, 
                 timeline, 
                 name="SPD", 
                 qe=0.8, 
                 dark_count_rate=1e3, 
                 jitter_mean=50e-12, 
                 jitter_std=10e-12, 
                 dead_time=100e-9, 
                 sensitivity_threshold=1e-12):
        """
        Initializes the Single Photon Detector.

        Args:
            timeline (Timeline): Simulation timeline for scheduling events.
            name (str): Detector name.
            qe (float): Quantum efficiency (0 to 1).
            dark_count_rate (float): Dark count rate (Hz).
            jitter_mean (float): Mean detector jitter (seconds).
            jitter_std (float): Standard deviation of jitter (seconds).
            dead_time (float): Detector dead time after a detection (seconds).
            sensitivity_threshold (float): Minimum power required to detect a photon.
        """
        self.timeline = timeline
        self.name = name
        self.qe = qe
        self.dark_count_rate = dark_count_rate
        self.jitter_mean = jitter_mean
        self.jitter_std = jitter_std
        self.dead_time = dead_time
        self.sensitivity_threshold = sensitivity_threshold
        self.last_detection_time = -np.inf  # Last detection event time

        # Schedule dark counts
        self.schedule_dark_counts()

    def detect_photon(self, event_time, power, Ex, Ey, E):
        """
        Handles the detection of an incoming photon.

        Args:
            event_time (float): Timestamp of the photon arrival.
            power (float): Optical power of the photon.
            Ex (float): Electric field component along x-axis.
            Ey (float): Electric field component along y-axis.
            E (float): Total electric field magnitude.
        """
        if power < self.sensitivity_threshold:
            print(f"[{event_time:.9f} s] {self.name}: Photon below threshold. No detection.")
            return

        # Check for dead time (if too soon after the last detection, ignore)
        if event_time - self.last_detection_time < self.dead_time:
            print(f"[{event_time:.9f} s] {self.name}: Detector in dead time. Photon ignored.")
            return

        # Compute detection probability
        detection_prob = self.qe * np.random.rand()

        if detection_prob > self.dark_count_rate * self.timeline.dt:
            # Introduce jitter (Gaussian delay)
            detection_delay = np.random.normal(self.jitter_mean, self.jitter_std)
            detection_time = event_time + detection_delay

            print(f"[{detection_time:.9f} s] {self.name}: Photon detected!")

            # Schedule a detection event
            self.timeline.schedule_event(detection_delay, self.register_detection, detection_time)

            # Update last detection time
            self.last_detection_time = detection_time
        else:
            print(f"[{event_time:.9f} s] {self.name}: Photon missed.")

    def register_detection(self, detection_time):
        """
        Registers a valid photon detection event.
        
        Args:
            detection_time (float): Timestamp of detection.
        """
        print(f"[{detection_time:.9f} s] {self.name}: Detection event registered.")

    def generate_dark_count(self):
        """
        Simulates dark counts due to thermal noise and background radiation.
        """
        # Introduce random jitter for dark count detection time
        jittered_time = time.time() + np.random.exponential(1 / self.dark_count_rate)

        # Schedule the dark count event
        self.timeline.schedule_event(jittered_time - time.time(), self.register_detection, jittered_time)

        print(f"[{jittered_time:.9f} s] {self.name}: Dark count event generated.")

        # Reschedule next dark count
        self.schedule_dark_counts()

    def schedule_dark_counts(self):
        """
        Continuously schedules dark count events.
        """
        interval = np.random.exponential(1 / self.dark_count_rate)  # Random interval based on Poisson process
        self.timeline.schedule_event(interval, self.generate_dark_count)


