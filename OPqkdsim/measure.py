import numpy as np
import time
from .timeline import Timeline

class SinglePhotonDetector:
    """
    Simulates a Single Photon Detector (SPD)
    
    Models detection probability using Poisson distribution, dark counts, after-pulsing, and dead time.
    """

    def __init__(self, 
                 timeline, 
                 name="SPD", 
                 qe=0.8, 
                 dark_count_rate=1e3, 
                 jitter_mean=50e-12, 
                 jitter_std=10e-12, 
                 dead_time=100e-9, 
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

        # Compute Pclick using Eq. (21) with proper probabilities
        Pclick = (Pp + self.after_pulsing_prob + Pd 
                  - Pp * self.after_pulsing_prob 
                  - self.after_pulsing_prob * Pd 
                  - Pd * Pp 
                  + Pp * self.after_pulsing_prob * Pd)
        rand = np.random.rand()
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
