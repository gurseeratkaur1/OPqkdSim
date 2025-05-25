from django.shortcuts import render
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ThOPqkdsim.sim7 import *
from ThOPqkdsim.sim8 import *
from ThOPqkdsim.sim9 import *
from ThOPqkdsim.sim10_5 import *
# Create your views here.

def home(request):
    return render(request, "home.html", {})

def get_plot_base64(plt):
    """
    Convert a matplotlib figure to a base64 encoded string for embedding in HTML
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Encode the PNG image to base64 string
    return base64.b64encode(image_png).decode('utf-8')


def modified_plot_qber_vs_mu(mu_values, time_window, distance,
                      alice_detector_efficiency, bob_detector_efficiency,
                      alice_channel_base_efficiency, bob_channel_base_efficiency,
                      dark_count_rate):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values
    """
    qber_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance
        )
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Mean Photon Number')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values

def modified_plot_skr_vs_mu(mu_values, time_window, key_length, distance,
                     alice_detector_efficiency, bob_detector_efficiency,
                     alice_channel_base_efficiency, bob_channel_base_efficiency,
                     dark_count_rate, repetition_rate):
    """
    Modified version of plot_skr_vs_mu that returns the base64 encoded plot
    and the SKR values
    """
    skr_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance
        )
        skr_per_pulse = simulator.calculate_skr(key_length)
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title('Secret Key Rate vs Mean Photon Number')
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values

def modified_plot_qber_vs_distance(distance_values, time_window, mu,
                           alice_detector_efficiency, bob_detector_efficiency,
                           alice_channel_base_efficiency, bob_channel_base_efficiency,
                           dark_count_rate):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot
    """
    qber_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'ro-', linewidth=2)
    plt.grid(True)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.axhline(y=11, color='red', linestyle='--', label='11% limit')
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Distance')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64

def modified_plot_skr_vs_distance(distance_values, time_window, key_length, mu,
                          alice_detector_efficiency, bob_detector_efficiency,
                          alice_channel_base_efficiency, bob_channel_base_efficiency,
                          dark_count_rate,repetition_rate):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot
    """
    skr_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        skr_per_pulse = simulator.calculate_skr(key_length)
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'mo-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title('Secret Key Rate vs Distance')
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64



def bb84(request):
    """
    View function to handle the BB84 simulator form and run simulations.
    """
    if request.method == 'POST':
        # Get form data
        mu = float(request.POST.get('mu', 0.1))
        distance = float(request.POST.get('distance', 10))
        attenuation = float(request.POST.get('attenuation', 0.2))
        alice_detector_efficiency = float(request.POST.get('alice_detector_efficiency', 0.8))
        bob_detector_efficiency = float(request.POST.get('bob_detector_efficiency', 0.8))
        alice_channel_base_efficiency = float(request.POST.get('alice_channel_base_efficiency', 1.0))
        bob_channel_base_efficiency = float(request.POST.get('bob_channel_base_efficiency', 0.3913))
        dark_count_rate = float(request.POST.get('dark_count_rate', 1000))
        time_window = float(request.POST.get('time_window', 1)) * 1e-9  # Convert ns to seconds
        repetition_rate = float(request.POST.get('repetition_rate', 1000000))
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', 0.01))
        mu_max = float(request.POST.get('mu_max', 1.0))
        distance_max_qber = float(request.POST.get('distance_max_qber', 200))
        distance_max_skr = float(request.POST.get('distance_max_skr', 150))
        
        # Define plot ranges
        mu_values = np.linspace(mu_min, mu_max, 10)
        distance_values_qber = np.linspace(0, distance_max_qber, 10)
        distance_values_skr = np.linspace(0, distance_max_skr, 10)
        
        # Generate plots
        qber_vs_mu_plot, qber_values = modified_plot_qber_vs_mu(
            mu_values, time_window, distance,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate
        )
        
        skr_vs_mu_plot, _ = modified_plot_skr_vs_mu(
            mu_values, time_window, 1000000, distance,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, repetition_rate
        )
        
        # Identify optimal μ value
        optimal_indices = [i for i, qber in enumerate(qber_values) if 5 <= qber <= 7]
        if optimal_indices:
            optimal_mu_index = optimal_indices[len(optimal_indices)//2]
            optimal_mu = mu_values[optimal_mu_index]
        else:
            optimal_mu_index = np.argmin(np.abs(np.array(qber_values) - 6))
            optimal_mu = mu_values[optimal_mu_index]
        
        # Generate distance plots with optimal μ
        qber_vs_distance_plot = modified_plot_qber_vs_distance(
            distance_values_qber, time_window, optimal_mu,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate
        )
        
        skr_vs_distance_plot = modified_plot_skr_vs_distance(
            distance_values_skr, time_window, 10000, optimal_mu,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, repetition_rate
        )
        
        # Calculate additional metrics
        simulator = BB84Simulator(
            mu=optimal_mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            attenuation=attenuation
        )
        
        optimal_qber = simulator.calculate_qber()
        optimal_skr = simulator.calculate_skr(1000)
        
        # Find max distance where QBER ≤ 11%
        max_distance = 0
        for d in np.arange(0, distance_max_qber, 10):
            simulator.update_distance(d)
            qber = simulator.calculate_qber()
            if qber <= 11:
                max_distance = d
            else:
                break
        
        # Package results for the template
        plots = {
            'qber_vs_mu': qber_vs_mu_plot,
            'skr_vs_mu': skr_vs_mu_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'skr_vs_distance': skr_vs_distance_plot
        }
        
        return render(request, 'bbm92.html', {
            'plots': plots,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_qber': f"{optimal_qber:.4f}",
            'optimal_skr': f"{optimal_skr:.6f}",
            'max_distance': f"{max_distance:.1f}",
            'repetition_rate': repetition_rate 
        })
    
    # If GET request, just render the form
    return render(request, 'bbm92.html')


def plot_key_rate_vs_distance_base64(qkd, max_distance=150):
    """Plot key rate vs distance and return base64 encoded image"""
    distances = np.arange(0, max_distance + 1, 1)
    key_rates = []
    
    for d in distances:
        rate, _, _, _, _, _ = qkd.key_rate(d)
        key_rates.append(rate)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(distances, key_rates)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secure Key Rate (bits/s)')
    plt.title('Secure Key Rate vs Distance')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt)

def plot_qber_vs_distance_base64(qkd, max_distance=150):
    """Plot QBER vs distance and return base64 encoded image"""
    distances = np.arange(0, max_distance + 1, 1)
    qbers = []
    
    for d in distances:
        _, qber, _, _, _, _ = qkd.key_rate(d)
        qbers.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, [qber * 100 for qber in qbers])  # Convert to percentage
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title('QBER vs Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt)

def plot_key_rate_vs_mu_base64(qkd, distance=50, mu_range=(0.1, 1.0), step=0.05):
    """Plot key rate vs signal state intensity and return base64 encoded image"""
    mu_values = np.arange(mu_range[0], mu_range[1] + step, step)
    key_rates = []
    
    original_mu = qkd.mu
    for mu in mu_values:
        qkd.mu = mu
        rate, _, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
    
    qkd.mu = original_mu
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, key_rates)
    plt.xlabel('Signal State Intensity (μ)')
    plt.ylabel('Secure Key Rate (bits/s)')
    plt.title(f'Secure Key Rate vs Signal State Intensity at {distance} km')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt)

def plot_key_rate_vs_nu1_base64(qkd, distance=50, nu1_range=(0.01, 0.3), step=0.02):
    """Plot key rate vs decoy state intensity nu1 and return base64 encoded image"""
    nu1_values = np.arange(nu1_range[0], nu1_range[1] + step, step)
    key_rates = []
    
    original_nu1 = qkd.nu1
    for nu1 in nu1_values:
        qkd.nu1 = nu1
        rate, _, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
    
    qkd.nu1 = original_nu1
    
    plt.figure(figsize=(10, 6))
    plt.plot(nu1_values, key_rates)
    plt.xlabel('Decoy State Intensity (ν₁)')
    plt.ylabel('Secure Key Rate (bits/s)')
    plt.title(f'Secure Key Rate vs Decoy State Intensity at {distance} km')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt)

def plot_qber_vs_mu_base64(qkd, distance=50, mu_range=(0.1, 3), step=0.01):
    """Plot QBER vs signal state intensity (mu) and return base64 encoded image"""
    # Calculate transmittance
    eta = qkd.calculate_total_transmittance(distance)
    
    # Create arrays for plotting
    mu_values = np.arange(mu_range[0], mu_range[1] + step, step)
    qber_values = []
    
    # Scientific formulation based on Lo et al. (2005) and Ma et al. (2005)
    for mu in mu_values:
        # Calculate probabilities of different photon number states
        p_vacuum = np.exp(-mu)  # Probability of vacuum state
        p_single = mu * np.exp(-mu)  # Probability of single-photon state
        p_multi = 1 - p_vacuum - p_single  # Probability of multi-photon states
        
        # Calculate detection probabilities
        # Vacuum state: only dark counts contribute
        Y_vacuum = qkd.Y0
        
        # Single photon state: combination of signal detection and dark counts
        Y_single = eta + qkd.Y0 - eta * qkd.Y0
        
        # Multi-photon states: higher detection probability
        Y_multi = 1 - (1-eta)**2 + qkd.Y0 - qkd.Y0 * (1-(1-eta)**2)
        
        # Calculate gains for each component
        Q_vacuum = p_vacuum * Y_vacuum
        Q_single = p_single * Y_single
        Q_multi = p_multi * Y_multi
        
        # Calculate error rates for each component
        E_vacuum = 0.5  # Random errors for vacuum (dark counts)
        E_single = qkd.e_detector
        
        # For multi-photon states, error increases due to information leakage
        # This is the key scientific principle that causes QBER to increase with mu
        E_multi = qkd.e_detector * (1 + 0.1 * mu)  # Error increases with mu
        
        # Calculate overall QBER using weighted average
        total_gain = Q_vacuum + Q_single + Q_multi
        total_error = (Q_vacuum * E_vacuum + Q_single * E_single + Q_multi * E_multi)
        
        qber = total_error / total_gain if total_gain > 0 else 0
        qber_values.append(qber * 100)  # Convert to percentage
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values)
    plt.xlabel('Signal State Intensity (μ)')
    plt.ylabel('QBER (%)')
    plt.title(f'QBER vs Signal State Intensity at {distance} km')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt)

def decoy_bb84(request):
    """
    View function to handle the Decoy State QKD simulator form and run simulations.
    """
    if request.method == 'POST':
        # Extract form data with defaults
        wavelength = float(request.POST.get('wavelength', 1550))
        alpha = float(request.POST.get('alpha', 0.21))
        e_detector = float(request.POST.get('e_detector', 0.033))
        Y0 = float(request.POST.get('Y0', 1.7e-6))
        eta_bob = float(request.POST.get('eta_bob', 0.045))
        mu = float(request.POST.get('mu', 0.5))
        nu1 = float(request.POST.get('nu1', 0.1))
        nu2 = float(request.POST.get('nu2', 0.0))
        f = float(request.POST.get('f', 1.22))
        q = float(request.POST.get('q', 0.5))
        rep_rate = float(request.POST.get('rep_rate', 2e6))
        
        # Get plot configuration parameters
        max_distance = float(request.POST.get('max_distance', 150))
        selected_distance = float(request.POST.get('selected_distance', 50))
        mu_min = float(request.POST.get('mu_min', 0.1))
        mu_max = float(request.POST.get('mu_max', 3.0))
        mu_step = float(request.POST.get('mu_step', 0.05))
        nu1_min = float(request.POST.get('nu1_min', 0.01))
        nu1_max = float(request.POST.get('nu1_max', 0.3))
        nu1_step = float(request.POST.get('nu1_step', 0.02))
        
        # Initialize QKD simulator
        qkd = DecoyStateQKD(
            wavelength=wavelength, 
            alpha=alpha,
            e_detector=e_detector, 
            Y0=Y0,
            eta_bob=eta_bob, 
            mu=mu,
            nu1=nu1, 
            nu2=nu2,
            f=f, 
            q=q,
            rep_rate=rep_rate
        )
        
        # Generate plots
        key_rate_distance_plot = plot_key_rate_vs_distance_base64(qkd, max_distance)
        qber_distance_plot = plot_qber_vs_distance_base64(qkd, max_distance)
        key_rate_mu_plot = plot_key_rate_vs_mu_base64(qkd, selected_distance, 
                                                    (mu_min, mu_max), mu_step)
        key_rate_nu1_plot = plot_key_rate_vs_nu1_base64(qkd, selected_distance, 
                                                      (nu1_min, nu1_max), nu1_step)
        qber_mu_plot = plot_qber_vs_mu_base64(qkd, selected_distance, 
                                           (mu_min, mu_max), mu_step)
        
        # Calculate current settings performance
        current_rate, current_qber, current_gain, Y1_L, e1_U, Q1_L = qkd.key_rate(selected_distance)
        
        # Find maximum achievable distance
        max_achievable_distance = 0
        for d in range(0, 300, 5):  # Check every 5km up to 300km
            rate, _, _, _, _, _ = qkd.key_rate(d)
            if rate > 0:
                max_achievable_distance = d
            else:
                break
        
        # Find optimal mu value at selected distance
        mu_values = np.arange(mu_min, mu_max + mu_step, mu_step)
        mu_rates = []
        original_mu = qkd.mu
        
        for m in mu_values:
            qkd.mu = m
            rate, _, _, _, _, _ = qkd.key_rate(selected_distance)
            mu_rates.append(rate)
        
        qkd.mu = original_mu
        optimal_mu_index = np.argmax(mu_rates)
        optimal_mu = mu_values[optimal_mu_index]
        optimal_rate = mu_rates[optimal_mu_index]
        
        # Package results for the template
        plots = {
            'key_rate_distance': key_rate_distance_plot,
            'qber_distance': qber_distance_plot,
            'key_rate_mu': key_rate_mu_plot,
            'key_rate_nu1': key_rate_nu1_plot,
            'qber_mu': qber_mu_plot,
        }
        
        return render(request, 'decoybb84.html', {
            'plots': plots,
            'current_rate': f"{current_rate:.4f}",
            'current_qber': f"{current_qber * 100:.4f}",
            'current_gain': f"{current_gain:.6f}",
            'Y1_L': f"{Y1_L:.6f}",
            'e1_U': f"{e1_U:.6f}",
            'Q1_L': f"{Q1_L:.6f}",
            'max_achievable_distance': f"{max_achievable_distance}",
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_rate': f"{optimal_rate:.6f}",
            'form_data': request.POST,  # Pass form data back to populate the form
        })
    
    # If GET request, just render the form with default values
    return render(request, 'decoybb84.html')

def cow_modified_plot_qber_vs_mu(mu_values, distance, 
                             detector_efficiency, dark_count_rate,
                             time_window, channel_base_efficiency,
                             attenuation, data_line_ratio,
                             decoy_probability, repetition_rate):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values
    """
    qber_values = []
    
    # Create simulator
    simulator = COWProtocol(
        distance=distance,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate QBER for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'b-', linewidth=2)
    plt.plot(mu_values, qber_values, 'bo', markersize=6)
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('QBER (%)', fontsize=12)
    plt.title(f'Quantum Bit Error Rate vs Mean Photon Number in COW Protocol (Distance: {distance} km)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values


def cow_modified_plot_skr_vs_mu(mu_values, distance, 
                            detector_efficiency, dark_count_rate,
                            time_window, channel_base_efficiency,
                            attenuation, data_line_ratio,
                            decoy_probability, repetition_rate):
    """
    Modified version of plot_skr_vs_mu that returns the base64 encoded plot
    and the SKR values
    """
    skr_values = []
    
    # Create simulator
    simulator = COWProtocol(
        distance=distance,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate SKR for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'g-', linewidth=2)
    plt.plot(mu_values, skr_values, 'go', markersize=6)
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=12)
    plt.title(f'Secret Key Rate vs Mean Photon Number in COW Protocol (Distance: {distance} km)', fontsize=14)
    
    # Mark optimal mean photon number
    if max(skr_values) > 0:
        optimal_mu_index = np.argmax(skr_values)
        optimal_mu = mu_values[optimal_mu_index]
        optimal_skr = skr_values[optimal_mu_index]
        
        plt.plot(optimal_mu, optimal_skr, 'ro', markersize=8)
        plt.annotate(f'Optimal μ ≈ {optimal_mu:.2f}\nSKR ≈ {optimal_skr:.2e} bits/s',
                     xy=(optimal_mu, optimal_skr), 
                     xytext=(optimal_mu + 0.1, optimal_skr * 0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values


def cow_modified_plot_qber_vs_distance(distance_values, mu, 
                                  detector_efficiency, dark_count_rate,
                                  time_window, channel_base_efficiency,
                                  attenuation, data_line_ratio,
                                  decoy_probability, repetition_rate):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot
    """
    qber_values = []
    
    # Create simulator
    simulator = COWProtocol(
        mu=mu,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate QBER for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'r-', linewidth=2)
    plt.plot(distance_values, qber_values, 'ro', markersize=6)
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('QBER (%)', fontsize=12)
    plt.title(f'Quantum Bit Error Rate vs Distance in COW Protocol (μ = {mu})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64


def cow_modified_plot_skr_vs_distance(distance_values, mu, 
                                 detector_efficiency, dark_count_rate,
                                 time_window, channel_base_efficiency,
                                 attenuation, data_line_ratio,
                                 decoy_probability, repetition_rate):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot
    """
    skr_values = []
    
    # Create simulator
    simulator = COWProtocol(
        mu=mu,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate SKR for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'm-', linewidth=2)
    plt.plot(distance_values, skr_values, 'mo', markersize=6)
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=12)
    plt.title(f'Secret Key Rate vs Distance in COW Protocol (μ = {mu})', fontsize=14)
    
    # Find maximum distance with positive SKR
    positive_indices = np.where(np.array(skr_values) > 0)[0]
    if len(positive_indices) > 0:
        max_dist_idx = positive_indices[-1]
        max_dist = distance_values[max_dist_idx]
        plt.axvline(x=max_dist, color='r', linestyle=':', alpha=0.7)
        plt.text(max_dist + 5, max(skr_values)/10, f'Max distance: {max_dist:.1f} km', color='r')
    
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64


def cowqkd(request):
    """
    View function to handle the COW QKD simulator form and run simulations.
    """
    if request.method == 'POST':
        # Get form data
        mu = float(request.POST.get('mu', 0.5))
        distance = float(request.POST.get('distance', 50))
        attenuation = float(request.POST.get('attenuation', 0.2))
        detector_efficiency = float(request.POST.get('detector_efficiency', 0.2))
        dark_count_rate = float(request.POST.get('dark_count_rate', 500))
        time_window = float(request.POST.get('time_window', 1)) * 1e-9  # Convert ns to seconds
        channel_base_efficiency = float(request.POST.get('channel_base_efficiency', 0.8))
        data_line_ratio = float(request.POST.get('data_line_ratio', 0.9))
        decoy_probability = float(request.POST.get('decoy_probability', 0.1))
        repetition_rate = float(request.POST.get('repetition_rate', 500e6))
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', 0.01))
        mu_max = float(request.POST.get('mu_max', 1.0))
        distance_max_qber = float(request.POST.get('distance_max_qber', 200))
        distance_max_skr = float(request.POST.get('distance_max_skr', 200))
        
        # Define plot ranges
        mu_values = np.linspace(mu_min, mu_max, 20)
        distance_values_qber = np.linspace(0, distance_max_qber, 50)
        distance_values_skr = np.linspace(0, distance_max_skr, 50)
        
        # Generate plots
        qber_vs_mu_plot, qber_values = cow_modified_plot_qber_vs_mu(
            mu_values, distance,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        skr_vs_mu_plot, skr_values = cow_modified_plot_skr_vs_mu(
            mu_values, distance,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        # Identify optimal μ value for maximum SKR
        if max(skr_values) > 0:
            optimal_mu_index = np.argmax(skr_values)
            optimal_mu = mu_values[optimal_mu_index]
        else:
            # If no positive SKR found, use a default value for μ
            optimal_mu = 0.3  # A common value for COW protocol
        
        # Generate distance plots with optimal μ
        qber_vs_distance_plot = cow_modified_plot_qber_vs_distance(
            distance_values_qber, optimal_mu,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        skr_vs_distance_plot = cow_modified_plot_skr_vs_distance(
            distance_values_skr, optimal_mu,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        # Calculate additional metrics
        simulator = COWProtocol(
            mu=optimal_mu,
            distance=distance,
            detector_efficiency=detector_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            channel_base_efficiency=channel_base_efficiency,
            attenuation=attenuation,
            data_line_ratio=data_line_ratio,
            decoy_probability=decoy_probability,
            repetition_rate=repetition_rate
        )
        
        optimal_qber = simulator.calculate_qber()
        optimal_skr = simulator.calculate_skr()
        
        # Find max distance where QBER ≤ 11% (security threshold for COW)
        max_distance = 0
        for d in np.arange(0, 300, 1):
            simulator.update_distance(d)
            qber = simulator.calculate_qber()
            if qber <= 11:
                max_distance = d
            else:
                break
        
        # Package results for the template
        plots = {
            'qber_vs_mu': qber_vs_mu_plot,
            'skr_vs_mu': skr_vs_mu_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'skr_vs_distance': skr_vs_distance_plot
        }
        
        return render(request, 'cowqkd.html', {
            'plots': plots,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_qber': f"{optimal_qber:.4f}",
            'optimal_skr': f"{optimal_skr:.4e}",
            'max_distance': f"{max_distance:.1f}",
            'repetition_rate': f"{repetition_rate:.2e}"
        })
    
    # If GET request, just render the form
    return render(request, 'cowqkd.html')


def dps_modified_plot_qber_vs_mu(simulator, mu_range, points=100):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values
    """
    mu_values, qber_values = simulator.get_qber_vs_mu_data(mu_range, points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'b-', linewidth=2)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('QBER', fontsize=12)
    plt.title('QBER vs Mean Photon Number', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, mu_values, qber_values

def dps_modified_plot_qber_vs_distance(simulator, distance_range, points=100):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot
    """
    distance_values, qber_values = simulator.get_qber_vs_distance_data(distance_range, points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'g-', linewidth=2)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('QBER', fontsize=12)
    plt.title('QBER vs Distance', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, distance_values, qber_values

def dps_modified_plot_skr_vs_mu(simulator, mu_range, points=100):
    """
    Modified version of plot_skr_vs_mu that returns the base64 encoded plot
    and the SKR values
    """
    mu_values, skr_values = simulator.get_skr_vs_mu_data(mu_range, points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'r-', linewidth=2)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('SKR (bits/s)', fontsize=12)
    plt.title('Secret Key Rate vs Mean Photon Number', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, mu_values, skr_values

def dps_modified_plot_skr_vs_distance(simulator, distance_range, points=100):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot
    """
    distance_values, skr_values = simulator.get_skr_vs_distance_data(distance_range, points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'c-', linewidth=2)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('SKR (bits/s)', fontsize=12)
    plt.title('Secret Key Rate vs Distance', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, distance_values, skr_values

def dpsqkd(request):
    """
    View function to handle the DPS QKD simulator form and run simulations.
    """
    if request.method == 'POST':
        # Get form data
        nem = int(float(request.POST.get('nem', 1e6)))
        repetition_rate = float(request.POST.get('repetition_rate', 1e9))
        mu = float(request.POST.get('mu', 0.1))
        ebit = float(request.POST.get('ebit', 0.01))
        eta_det = float(request.POST.get('eta_det', 0.1))
        dark_count = float(request.POST.get('dark_count', 1e-7))
        alpha = float(request.POST.get('alpha', 0.2))
        distance = float(request.POST.get('distance', 50))
        delta_bs1 = float(request.POST.get('delta_bs1', 0.005))
        delta_bs2 = float(request.POST.get('delta_bs2', 0.005))
        t_prob = float(request.POST.get('t_prob', 0.5))
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', 0.001))
        mu_max = float(request.POST.get('mu_max', 0.5))
        distance_max = float(request.POST.get('distance_max', 200))
        
        # Define plot ranges
        mu_range = (mu_min, mu_max)
        distance_range = (0, distance_max)
        
        # Create the simulator with the parameters
        simulator = DPSQKDSimulator(
            nem=nem,
            repetition_rate=repetition_rate,
            mu=mu,
            ebit=ebit,
            eta_det=eta_det,
            dark_count=dark_count,
            alpha=alpha,
            distance=distance,
            delta_bs1=delta_bs1,
            delta_bs2=delta_bs2,
            t_prob=t_prob
        )
        
        # Generate plots
        qber_vs_mu_plot, mu_values, qber_values = dps_modified_plot_qber_vs_mu(
            simulator, mu_range, points=100
        )
        
        skr_vs_mu_plot, _, skr_mu_values = dps_modified_plot_skr_vs_mu(
            simulator, mu_range, points=100
        )
        
        qber_vs_distance_plot, _, _ = dps_modified_plot_qber_vs_distance(
            simulator, distance_range, points=100
        )
        
        skr_vs_distance_plot, distance_values, skr_distance_values = dps_modified_plot_skr_vs_distance(
            simulator, distance_range, points=100
        )
        
        # Find optimal μ value (highest key rate)
        optimal_mu_index = np.argmax(skr_mu_values)
        optimal_mu = mu_values[optimal_mu_index]
        
        # Find maximum achievable distance (last positive SKR)
        max_distance = 0
        for i, skr in enumerate(skr_distance_values):
            if skr > 0:
                max_distance = distance_values[i]
        
        # Calculate current metrics with the given parameters
        current_qber = simulator.calculate_qber()
        current_skr = simulator.calculate_secret_key_rate()
        
        # Package results for the template
        plots = {
            'qber_vs_mu': qber_vs_mu_plot,
            'skr_vs_mu': skr_vs_mu_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'skr_vs_distance': skr_vs_distance_plot
        }
        
        return render(request, 'dpsqkd.html', {
            'plots': plots,
            'nem': nem,
            'repetition_rate': repetition_rate,
            'mu': mu,
            'ebit': ebit,
            'eta_det': eta_det,
            'dark_count': dark_count,
            'alpha': alpha,
            'distance': distance,
            'delta_bs1': delta_bs1,
            'delta_bs2': delta_bs2,
            't_prob': t_prob,
            'optimal_mu': f"{optimal_mu:.4f}",
            'max_distance': f"{max_distance:.1f}",
            'current_qber': f"{current_qber:.6f}",
            'current_skr': f"{current_skr:.2f}",
        })
    
    # If GET request, just render the form with default values
    return render(request, 'dpsqkd.html', {
        'nem': 1e6,
        'repetition_rate': 1e9,
        'mu': 0.1,
        'ebit': 0.01,
        'eta_det': 0.1, 
        'dark_count': 1e-7,
        'alpha': 0.2,
        'distance': 50,
        'delta_bs1': 0.005,
        'delta_bs2': 0.005,
        't_prob': 0.5,
        'mu_min': 0.001,
        'mu_max': 0.5,
        'distance_max': 200
    })