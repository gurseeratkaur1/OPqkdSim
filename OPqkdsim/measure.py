import numpy as np
import time
from .timeline import Timeline

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
