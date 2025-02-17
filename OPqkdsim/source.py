import numpy as np
from scipy.integrate import solve_ivp
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
