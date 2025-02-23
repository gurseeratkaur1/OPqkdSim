import numpy as np
from .timeline import *
from .random import *

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
