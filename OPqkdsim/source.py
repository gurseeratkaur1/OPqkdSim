import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import poisson, chi2_contingency, norm
from .timeline import *

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
        eta_DFB=0.8,  # Differential quantum efficiency
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
        S_final = max(1, poisson.rvs(mu=self.mu))  # Ensure at least 1 photon # Sample photon number from Poisson distribution

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


