import numpy as np
from .timeline import *

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