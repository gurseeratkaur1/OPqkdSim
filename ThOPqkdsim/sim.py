import numpy as np

class BB84Simulator:
    def __init__(self, 
                 mu, 
                 distance, 
                 alpha_PBS, 
                 alpha_BS, 
                 eta, 
                 TAE, 
                 TEB, 
                 detector_efficiency, 
                 dark_count_rate, 
                 background_rate, 
                 dead_time, 
                 time_slot_duration):
        # System parameters
        self.mu = mu  # Photon number (brightness) parameter
        self.distance = distance  # Distance between Alice and Bob
        self.alpha_PBS = alpha_PBS  # Polarizing beam splitter efficiency
        self.alpha_BS = alpha_BS  # Beam splitter efficiency
        self.eta = eta  # Channel efficiency
        self.TAE = TAE  # Transmission from Alice to Eve
        self.TEB = TEB  # Transmission from Eve to Bob
        self.detector_efficiency = detector_efficiency  # Detector efficiency
        self.dark_count_rate = dark_count_rate  # Dark count rate in Hz/µm²
        self.background_rate = background_rate  # Background count rate in photons
        self.dead_time = dead_time  # Dead time after each detector click
        self.time_slot_duration = time_slot_duration  # Duration of each time slot
        
        # Derived system efficiencies
        self.eta_tx = alpha_PBS * alpha_BS * 0.5  # Transmission efficiency of the transmitter

    def simulate_channel(self, l):
        """
        Simulates the photon number distribution after passing through the channel
        l is the input number of photons
        """
        m = np.random.binomial(l, self.eta)
        return m

    def simulate_eavesdropper(self, l):
        """
        Simulates the effect of Eve's interception attack.
        l is the input number of photons
        """
        m = np.random.binomial(l, self.TAE)
        return m

    def simulate_detector(self, photons_received):
        """
        Simulate the detector clicks based on received photons.
        Considers detector efficiency, dark counts, and background noise.
        """
        # Calculate number of dark counts
        N_dark = self.time_slot_duration * (self.dark_count_rate * (self.detector_efficiency ** 2))
        P_dark = 1 - np.exp(-N_dark)
        
        # Calculate number of background counts
        P_back = 1 - np.exp(-self.background_rate)
        
        # Total noise probability
        P_noise = P_dark + P_back - P_dark * P_back
        
        # Simulate clicks
        clicks = np.random.binomial(photons_received, self.detector_efficiency)
        
        # Add noise
        noise_clicks = np.random.binomial(photons_received, P_noise)
        
        total_clicks = clicks + noise_clicks
        return total_clicks

    def simulate_BB84(self):
        """
        Simulate the full BB84 protocol with given parameters.
        """
        # Alice sends photons
        l = np.random.poisson(self.mu)  # Poisson distribution for number of photons sent by Alice
        photons_sent = self.simulate_channel(l)
        
        # Eve intercepts the photons
        photons_after_eve = self.simulate_eavesdropper(photons_sent)
        
        # Bob receives the photons
        photons_received = self.simulate_channel(photons_after_eve)
        
        # Simulate detector clicks on Bob's side
        clicks = self.simulate_detector(photons_received)
        
        # Calculate QBER (Quantum Bit Error Rate)
        qber = self.calculate_QBER(clicks, photons_sent)
        
        # Calculate SKR (Secret Key Rate)
        skr = self.calculate_SKR(clicks, photons_sent, qber)
        
        return qber, skr

    def calculate_QBER(self, clicks, photons_sent):
        """
        Calculate the Quantum Bit Error Rate (QBER).
        """
        errors = np.random.binomial(clicks, 0.25)  # 25% chance of error due to Eve's interference
        qber = errors / clicks if clicks > 0 else 0
        return qber

    def calculate_SKR(self, clicks, photons_sent, qber):
        """
        Calculate the Secret Key Rate (SKR) based on the QBER.
        """
        SKR = (clicks * (1 - qber)) / self.time_slot_duration  # Rate per time slot
        return SKR


