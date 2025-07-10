from django.shortcuts import render
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ThOPqkdsim.simBBM92 import *
from ThOPqkdsim.simDecoyState import *
from ThOPqkdsim.simCOW import *
from ThOPqkdsim.simDPS import *
from ThOPqkdsim.simBB84 import *
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
                      dark_count_rate, channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values
    """
    qber_values = []
    
    for mu in mu_values:
        simulator = BBM92Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            channel_mode=channel_mode
        )
        
        # Set FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('QBER (%)')
    
    channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    plt.title(f'Quantum Bit Error Rate vs Mean Photon Number ({channel_type})')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values

def modified_plot_skr_vs_mu(mu_values, time_window, key_length, distance,
                     alice_detector_efficiency, bob_detector_efficiency,
                     alice_channel_base_efficiency, bob_channel_base_efficiency,
                     dark_count_rate, repetition_rate, channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_skr_vs_mu that returns the base64 encoded plot
    and the SKR values
    """
    skr_values = []
    
    for mu in mu_values:
        simulator = BBM92Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            channel_mode=channel_mode
        )
        
        # Set FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        skr_per_pulse = simulator.calculate_skr(key_length)
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('Secret Key Rate (bits/s)')
    
    channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    plt.title(f'Secret Key Rate vs Mean Photon Number ({channel_type})')
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values

def modified_plot_qber_vs_distance(distance_values, time_window, mu,
                           alice_detector_efficiency, bob_detector_efficiency,
                           alice_channel_base_efficiency, bob_channel_base_efficiency,
                           dark_count_rate, channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot
    """
    qber_values = []
    
    simulator = BBM92Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,
        channel_mode=channel_mode
    )
    
    # Set FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
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
    
    channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    plt.title(f'Quantum Bit Error Rate vs Distance ({channel_type})')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values

def modified_plot_skr_vs_distance(distance_values, time_window, key_length, mu,
                          alice_detector_efficiency, bob_detector_efficiency,
                          alice_channel_base_efficiency, bob_channel_base_efficiency,
                          dark_count_rate, repetition_rate, channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot
    """
    skr_values = []
    
    simulator = BBM92Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,
        channel_mode=channel_mode
    )
    
    # Set FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
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
    
    channel_type = "Fiber Optic" if channel_mode == "fiber" else "Free Space Optical"
    plt.title(f'Secret Key Rate vs Distance ({channel_type})')
    plt.yscale('log')  # Log scale for better visualization
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values


def bbm92(request):
    """
    View function to handle the BBM92 simulator form and run simulations.
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
        
        # Get channel mode and FSO parameters
        channel_mode = request.POST.get('channel_mode', "fiber")
        
        # Get plot range parameters (adjust based on channel mode)
        mu_min = float(request.POST.get('mu_min', 0.01))
        mu_max = float(request.POST.get('mu_max', 1.0))
        
        # Set appropriate distance ranges based on channel mode
        if channel_mode == "fso":
            distance_max_qber = float(request.POST.get('fso_distance_max_qber', 100))
            distance_max_skr = float(request.POST.get('fso_distance_max_skr', 50))
        else:  # fiber mode
            distance_max_qber = float(request.POST.get('fiber_distance_max_qber', 300))
            distance_max_skr = float(request.POST.get('fiber_distance_max_skr', 300))

        # Define plot ranges
        mu_values = np.linspace(mu_min, mu_max, 20)
        distance_values_qber = np.linspace(0, distance_max_qber, 50)
        distance_values_skr = np.linspace(0, distance_max_skr, 50)
        
        # Process FSO parameters if in FSO mode
        fso_params = None
        if channel_mode == "fso":
            fso_params = {
                'transmitter_diameter': float(request.POST.get('transmitter_diameter', 0.1)),
                'receiver_diameter': float(request.POST.get('receiver_diameter', 0.3)),
                'beam_divergence': float(request.POST.get('beam_divergence', 0.001)),
                'wavelength': float(request.POST.get('wavelength', 850e-9)),
                'pointing_error': float(request.POST.get('pointing_error', 1e-6)),
                'transmitter_efficiency': float(request.POST.get('transmitter_efficiency', 0.9)),
                'receiver_efficiency': float(request.POST.get('receiver_efficiency', 0.9))
            }
        
        # Generate plots
        qber_vs_mu_plot, qber_values = modified_plot_qber_vs_mu(
            mu_values, time_window, distance,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, channel_mode, fso_params
        )
        
        skr_vs_mu_plot, _ = modified_plot_skr_vs_mu(
            mu_values, time_window, 1000000, distance,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, repetition_rate, channel_mode, fso_params
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
        qber_vs_distance_plot, _ = modified_plot_qber_vs_distance(
            distance_values_qber, time_window, optimal_mu,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, channel_mode, fso_params
        )
        
        skr_vs_distance_plot, _ = modified_plot_skr_vs_distance(
            distance_values_skr, time_window, 10000, optimal_mu,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, repetition_rate, channel_mode, fso_params
        )
        
        # Calculate additional metrics
        simulator = BBM92Simulator(
            mu=optimal_mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            attenuation=attenuation,
            channel_mode=channel_mode
        )
        
        # Set FSO parameters if needed
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
        
        optimal_qber = simulator.calculate_qber()
        optimal_skr = simulator.calculate_skr(1000)
        
        # Find max distance where QBER ≤ 11%
        max_distance = 0
        distance_step = 1 if channel_mode == "fso" else 10  # Smaller steps for FSO
        max_search_distance = 50 if channel_mode == "fso" else 300  # Shorter range for FSO
        
        for d in np.arange(0, max_search_distance, distance_step):
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
        
        context = {
            'plots': plots,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_qber': f"{optimal_qber:.4f}",
            'optimal_skr': f"{optimal_skr:.6f}",
            'max_distance': f"{max_distance:.1f}",
            'repetition_rate': repetition_rate,
            'channel_mode': channel_mode
        }
        
        # Add FSO parameters to context if using FSO mode
        if channel_mode == "fso" and fso_params is not None:
            context.update({
                'fso_params': fso_params
            })
        
        return render(request, 'bbm92.html', context)
    
    # If GET request, just render the form with default values
    context = {
        'default_params': {
            # Common parameters
            'mu': 0.1,
            'distance': 10,
            'attenuation': 0.2,
            'alice_detector_efficiency': 0.8,
            'bob_detector_efficiency': 0.8,
            'alice_channel_base_efficiency': 1.0,
            'bob_channel_base_efficiency': 0.3913,
            'dark_count_rate': 1000,
            'time_window': 1,  # in ns
            'repetition_rate': 1000000,
            'mu_min': 0.01,
            'mu_max': 1.0,
            
            # Fiber specific defaults
            'fiber_distance_max_qber': 300,
            'fiber_distance_max_skr': 300,
            
            # FSO specific defaults
            'fso_distance_max_qber': 100,
            'fso_distance_max_skr': 50,
            'transmitter_diameter': 0.1,  # in meters
            'receiver_diameter': 0.3,     # in meters
            'beam_divergence': 0.001,     # in radians (1 mrad)
            'wavelength': 850e-9,         # in meters (850 nm)
            'pointing_error': 1e-6,       # in radians
            'transmitter_efficiency': 0.9,
            'receiver_efficiency': 0.9
        }
    }
    
    return render(request, 'bbm92.html', context)

def plot_key_rate_vs_distance_base64(qkd, max_distance=150, channel_mode="fiber"):
    """Plot key rate vs distance and return base64 encoded image"""
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    distances = np.arange(0, max_distance + 1, 1)
    key_rates = []
    
    for d in distances:
        rate, _, _, _, _, _ = qkd.key_rate(d)
        key_rates.append(rate)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(distances, key_rates, 'mo-', linewidth=2, 
                 label=f"Secure Key Rate ({channel_mode.upper()})")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=14)
    plt.title(f'Secure Key Rate vs Distance - {channel_mode.upper()} Channel', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return get_plot_base64(plt)

def plot_qber_vs_distance_base64(qkd, max_distance=150, channel_mode="fiber"):
    """Plot QBER vs distance and return base64 encoded image"""
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    distances = np.arange(0, max_distance + 1, 1)
    qbers = []
    
    for d in distances:
        _, qber, _, _, _, _ = qkd.key_rate(d)
        qbers.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, [qber * 100 for qber in qbers], 'ro-', linewidth=2, 
             label=f"QBER ({channel_mode.upper()})")  # Convert to percentage
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('QBER (%)', fontsize=14)
    plt.title(f'QBER vs Distance - {channel_mode.upper()} Channel', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return get_plot_base64(plt)

def plot_key_rate_vs_mu_base64(qkd, distance=50, mu_range=(0.1, 1.0), step=0.05, channel_mode="fiber"):
    """Plot key rate vs signal state intensity and return base64 encoded image"""
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    mu_values = np.arange(mu_range[0], mu_range[1] + step, step)
    key_rates = []
    
    original_mu = qkd.mu
    for mu in mu_values:
        qkd.mu = mu
        rate, _, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
    
    qkd.mu = original_mu
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, key_rates, 'go-', linewidth=2, 
             label=f"Secure Key Rate ({channel_mode.upper()})")
    plt.xlabel('Signal State Intensity (μ)', fontsize=14)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=14)
    plt.title(f'Secure Key Rate vs Signal State Intensity at {distance} km - {channel_mode.upper()} Channel', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return get_plot_base64(plt)

def plot_key_rate_vs_nu1_base64(qkd, distance=50, nu1_range=(0.01, 0.3), step=0.02, channel_mode="fiber"):
    """Plot key rate vs decoy state intensity nu1 and return base64 encoded image"""
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    nu1_values = np.arange(nu1_range[0], nu1_range[1] + step, step)
    key_rates = []
    
    original_nu1 = qkd.nu1
    for nu1 in nu1_values:
        qkd.nu1 = nu1
        rate, _, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
    
    qkd.nu1 = original_nu1
    
    plt.figure(figsize=(10, 6))
    plt.plot(nu1_values, key_rates, 'bo-', linewidth=2, 
             label=f"Secure Key Rate ({channel_mode.upper()})")
    plt.xlabel('Decoy State Intensity (ν₁)', fontsize=14)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=14)
    plt.title(f'Secure Key Rate vs Decoy State Intensity at {distance} km - {channel_mode.upper()} Channel', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return get_plot_base64(plt)

def plot_qber_vs_mu_base64(qkd, distance=50, mu_range=(0.1, 3), step=0.01, channel_mode="fiber"):
    """Plot QBER vs signal state intensity (mu) and return base64 encoded image"""
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    # Create arrays for plotting
    mu_values = np.arange(mu_range[0], mu_range[1] + step, step)
    qber_values = []
    
    # Use the detailed QBER calculation method if available
    if hasattr(qkd, 'detailed_QBER'):
        # Calculate transmittance
        eta = qkd.calculate_total_transmittance(distance)
        
        for mu in mu_values:
            qber = qkd.detailed_QBER(mu, eta, distance)
            qber_values.append(qber * 100)  # Convert to percentage
    else:
        # Fallback to original implementation
        eta = qkd.calculate_total_transmittance(distance)
        
        for mu in mu_values:
            # Calculate probabilities of different photon number states
            p_vacuum = np.exp(-mu)  # Probability of vacuum state
            p_single = mu * np.exp(-mu)  # Probability of single-photon state
            p_multi = 1 - p_vacuum - p_single  # Probability of multi-photon states
            
            # Calculate detection probabilities
            Y_vacuum = qkd.Y0
            Y_single = eta + qkd.Y0 - eta * qkd.Y0
            Y_multi = 1 - (1-eta)**2 + qkd.Y0 - qkd.Y0 * (1-(1-eta)**2)
            
            # Calculate gains for each component
            Q_vacuum = p_vacuum * Y_vacuum
            Q_single = p_single * Y_single
            Q_multi = p_multi * Y_multi
            
            # Calculate error rates for each component
            E_vacuum = 0.5  # Random errors for vacuum (dark counts)
            E_single = qkd.e_detector
            E_multi = qkd.e_detector * (1 + 0.1 * mu)  # Error increases with mu
            
            # Calculate overall QBER using weighted average
            total_gain = Q_vacuum + Q_single + Q_multi
            total_error = (Q_vacuum * E_vacuum + Q_single * E_single + Q_multi * E_multi)
            
            qber = total_error / total_gain if total_gain > 0 else 0
            qber_values.append(qber * 100)  # Convert to percentage
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2, 
             label=f"QBER ({channel_mode.upper()})")
    plt.xlabel('Signal State Intensity (μ)', fontsize=14)
    plt.ylabel('QBER (%)', fontsize=14)
    plt.title(f'QBER vs Signal State Intensity at {distance} km - {channel_mode.upper()} Channel', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return get_plot_base64(plt)

def plot_key_rate_vs_error_base64(qkd, distance=50, error_range=(0.01, 0.1), step=0.005, channel_mode="fiber"):
    """Plot key rate vs detector error and return base64 encoded image"""
    # Set the channel mode
    qkd.set_channel_mode(channel_mode)
    
    error_values = np.arange(error_range[0], error_range[1] + step, step)
    key_rates = []
    
    original_error = qkd.e_detector
    for error in error_values:
        qkd.e_detector = error
        rate, _, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
    
    qkd.e_detector = original_error
    
    plt.figure(figsize=(10, 6))
    plt.plot(error_values, key_rates, 'bo-', linewidth=2, 
             label=f"Secure Key Rate ({channel_mode.upper()})")
    plt.xlabel('Detector Error Probability', fontsize=14)
    plt.ylabel('Secure Key Rate (bits/s)', fontsize=14)
    plt.title(f'Secure Key Rate vs Detector Error at {distance} km - {channel_mode.upper()} Channel', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return get_plot_base64(plt)

def decoy_bb84(request):
    """
    View function to handle the Decoy State QKD simulator form and run simulations.
    """
    # Default parameter values for each channel mode
    fiber_defaults = {
        'wavelength': 1550,     # nm
        'alpha': 0.21,          # dB/km
        'e_detector': 0.033,    # detector error probability (3.3%)
        'Y0': 1.7e-6,           # background rate
        'eta_bob': 0.045,       # Bob's side efficiency
        'mu': 0.5,              # signal state intensity
        'nu1': 0.1,             # decoy state 1 intensity
        'nu2': 0.0,             # decoy state 2 intensity (vacuum)
        'f': 1.22,              # error correction efficiency
        'q': 0.5,               # protocol efficiency factor (1/2 for BB84)
        'rep_rate': 1e6,        # repetition rate in Hz
        'max_distance': 150,    # Maximum distance for plots (km)
        'selected_distance': 50 # Default distance for analysis (km)
    }
    
    fso_defaults = {
        'wavelength': 1550,     # nm
        'alpha': 0.21,          # dB/km (same as fiber, but will be adjusted by FSO model)
        'e_detector': 0.033,    # detector error probability (3.3%)
        'Y0': 1.7e-6,           # background rate
        'eta_bob': 0.045,       # Bob's side efficiency
        'mu': 0.5,              # signal state intensity
        'nu1': 0.1,             # decoy state 1 intensity
        'nu2': 0.0,             # decoy state 2 intensity (vacuum)
        'f': 1.22,              # error correction efficiency
        'q': 0.5,               # protocol efficiency factor (1/2 for BB84)
        'rep_rate': 1e6,        # repetition rate in Hz
        'max_distance': 20,     # Maximum distance for plots (km) - shorter for FSO
        'selected_distance': 1.5  # Default distance for analysis (km) - shorter for FSO
    }
    
    if request.method == 'POST':
        # Get channel mode
        channel_mode = request.POST.get('channel_mode', 'fiber')
        
        # Select appropriate defaults based on channel mode
        defaults = fiber_defaults if channel_mode == 'fiber' else fso_defaults
        
        # Extract form data with defaults
        wavelength = float(request.POST.get('wavelength', defaults['wavelength']))
        alpha = float(request.POST.get('alpha', defaults['alpha']))
        e_detector = float(request.POST.get('e_detector', defaults['e_detector']))
        Y0 = float(request.POST.get('Y0', defaults['Y0']))
        eta_bob = float(request.POST.get('eta_bob', defaults['eta_bob']))
        mu = float(request.POST.get('mu', defaults['mu']))
        nu1 = float(request.POST.get('nu1', defaults['nu1']))
        nu2 = float(request.POST.get('nu2', defaults['nu2']))
        f = float(request.POST.get('f', defaults['f']))
        q = float(request.POST.get('q', defaults['q']))
        rep_rate = float(request.POST.get('rep_rate', defaults['rep_rate']))
        
        # Get plot configuration parameters
        max_distance = float(request.POST.get('max_distance', defaults['max_distance']))
        selected_distance = float(request.POST.get('selected_distance', defaults['selected_distance']))
        mu_min = float(request.POST.get('mu_min', 0.1))
        mu_max = float(request.POST.get('mu_max', 3.0))
        mu_step = float(request.POST.get('mu_step', 0.05))
        nu1_min = float(request.POST.get('nu1_min', 0.01))
        nu1_max = float(request.POST.get('nu1_max', 0.3))
        nu1_step = float(request.POST.get('nu1_step', 0.02))
        error_min = float(request.POST.get('error_min', 0.01))
        error_max = float(request.POST.get('error_max', 0.1))
        error_step = float(request.POST.get('error_step', 0.005))
        
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
            rep_rate=rep_rate,
            channel_mode=channel_mode
        )
        
        # Generate plots
        key_rate_distance_plot = plot_key_rate_vs_distance_base64(qkd, max_distance, channel_mode)
        qber_distance_plot = plot_qber_vs_distance_base64(qkd, max_distance, channel_mode)
        key_rate_mu_plot = plot_key_rate_vs_mu_base64(qkd, selected_distance, 
                                                    (mu_min, mu_max), mu_step, channel_mode)
        key_rate_nu1_plot = plot_key_rate_vs_nu1_base64(qkd, selected_distance, 
                                                      (nu1_min, nu1_max), nu1_step, channel_mode)
        qber_mu_plot = plot_qber_vs_mu_base64(qkd, selected_distance, 
                                           (mu_min, mu_max), mu_step, channel_mode)
        key_rate_error_plot = plot_key_rate_vs_error_base64(qkd, selected_distance,
                                                         (error_min, error_max), error_step, channel_mode)
        
        # Calculate current settings performance
        current_rate, current_qber, current_gain, Y1_L, e1_U, Q1_L = qkd.key_rate(selected_distance)
        
        # Find maximum achievable distance
        max_achievable_distance = 0
        max_check_distance = 300 if channel_mode == 'fiber' else 50  # Limit FSO search to 50km
        for d in range(0, max_check_distance, 5):  # Check every 5km
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
            'key_rate_error': key_rate_error_plot,
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
            'channel_mode': channel_mode,  # Pass channel mode back
        })
    
    # If GET request, just render the form with default values for fiber mode
    channel_mode = request.GET.get('channel_mode', 'fiber')
    defaults = fiber_defaults if channel_mode == 'fiber' else fso_defaults
    
    # Create a mock form_data to pass to the template
    form_data = {
        'channel_mode': channel_mode,
        'wavelength': defaults['wavelength'],
        'alpha': defaults['alpha'],
        'e_detector': defaults['e_detector'],
        'Y0': defaults['Y0'],
        'eta_bob': defaults['eta_bob'],
        'mu': defaults['mu'],
        'nu1': defaults['nu1'],
        'nu2': defaults['nu2'],
        'f': defaults['f'],
        'q': defaults['q'],
        'rep_rate': defaults['rep_rate'],
        'max_distance': defaults['max_distance'],
        'selected_distance': defaults['selected_distance'],
    }
    
    return render(request, 'decoybb84.html', {
        'form_data': form_data,
        'channel_mode': channel_mode,
    })


def cow_modified_plot_qber_vs_mu(mu_values, distance, 
                             detector_efficiency, dark_count_rate,
                             time_window, channel_base_efficiency,
                             attenuation, data_line_ratio,
                             decoy_probability, repetition_rate,
                             channel_mode="fiber"):
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
        repetition_rate=repetition_rate,
        channel_mode=channel_mode
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
    plt.title(f'Quantum Bit Error Rate vs Mean Photon Number in COW Protocol ({channel_mode.upper()}, Distance: {distance} km)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values


def cow_modified_plot_skr_vs_mu(mu_values, distance, 
                            detector_efficiency, dark_count_rate,
                            time_window, channel_base_efficiency,
                            attenuation, data_line_ratio,
                            decoy_probability, repetition_rate,
                            channel_mode="fiber"):
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
        repetition_rate=repetition_rate,
        channel_mode=channel_mode
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
    plt.title(f'Secret Key Rate vs Mean Photon Number in COW Protocol ({channel_mode.upper()}, Distance: {distance} km)', fontsize=14)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values


def cow_modified_plot_qber_vs_distance(distance_values, mu, 
                                  detector_efficiency, dark_count_rate,
                                  time_window, channel_base_efficiency,
                                  attenuation, data_line_ratio,
                                  decoy_probability, repetition_rate,
                                  channel_mode="fiber"):
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
        repetition_rate=repetition_rate,
        channel_mode=channel_mode
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
    plt.title(f'Quantum Bit Error Rate vs Distance in COW Protocol ({channel_mode.upper()}, μ = {mu})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64


def cow_modified_plot_skr_vs_distance(distance_values, mu, 
                                 detector_efficiency, dark_count_rate,
                                 time_window, channel_base_efficiency,
                                 attenuation, data_line_ratio,
                                 decoy_probability, repetition_rate,
                                 channel_mode="fiber"):
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
        repetition_rate=repetition_rate,
        channel_mode=channel_mode
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
    plt.title(f'Secret Key Rate vs Distance in COW Protocol ({channel_mode.upper()}, μ = {mu})', fontsize=14)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64


def cowqkd(request):
    """
    View function to handle the COW QKD simulator form and run simulations.
    """
    if request.method == 'POST':
        # Get channel mode first to determine appropriate default ranges
        channel_mode = request.POST.get('channel_mode', 'fiber')
        
        # Get form data
        mu = float(request.POST.get('mu', 0.5 if channel_mode == 'fiber' else 0.3))
        distance = float(request.POST.get('distance', 50 if channel_mode == 'fiber' else 5))
        attenuation = float(request.POST.get('attenuation', 0.2))
        detector_efficiency = float(request.POST.get('detector_efficiency', 0.2))
        dark_count_rate = float(request.POST.get('dark_count_rate', 500))
        time_window = float(request.POST.get('time_window', 1)) * 1e-9  # Convert ns to seconds
        channel_base_efficiency = float(request.POST.get('channel_base_efficiency', 0.8))
        data_line_ratio = float(request.POST.get('data_line_ratio', 0.9))
        decoy_probability = float(request.POST.get('decoy_probability', 0.1))
        repetition_rate = float(request.POST.get('repetition_rate', 500e6))
        
        # Get plot range parameters with defaults based on channel mode
        mu_min = float(request.POST.get('mu_min', 0.01))
        mu_max = float(request.POST.get('mu_max', 1.0))
        
        # Different distance ranges for fiber vs FSO
        if channel_mode == 'fiber':
            distance_max_qber = float(request.POST.get('distance_max_qber', 200))
            distance_max_skr = float(request.POST.get('distance_max_skr', 200))
        else:  # FSO mode
            distance_max_qber = float(request.POST.get('distance_max_qber', 50))
            distance_max_skr = float(request.POST.get('distance_max_skr', 25))
        
        # Define plot ranges
        mu_values = np.linspace(mu_min, mu_max, 50 if channel_mode == 'fso' else 20)
        distance_values_qber = np.linspace(0, distance_max_qber, 50)
        distance_values_skr = np.linspace(0, distance_max_skr, 50)
        
        # Generate plots
        qber_vs_mu_plot, qber_values = cow_modified_plot_qber_vs_mu(
            mu_values, distance,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate,
            channel_mode
        )
        
        skr_vs_mu_plot, skr_values = cow_modified_plot_skr_vs_mu(
            mu_values, distance,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate,
            channel_mode
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
            decoy_probability, repetition_rate,
            channel_mode
        )
        
        skr_vs_distance_plot = cow_modified_plot_skr_vs_distance(
            distance_values_skr, optimal_mu,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate,
            channel_mode
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
            repetition_rate=repetition_rate,
            channel_mode=channel_mode
        )
        
        optimal_qber = simulator.calculate_qber()
        optimal_skr = simulator.calculate_skr()
        
        # Find max distance where QBER ≤ 11% (security threshold for COW)
        # Set different max search distances based on channel mode
        max_search_distance = 300 if channel_mode == 'fiber' else 50
        max_distance = 0
        for d in np.arange(0, max_search_distance, 1):
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
            'repetition_rate': f"{repetition_rate:.2e}",
            'channel_mode': channel_mode
        })
    
    # If GET request, just render the form with default values
    # Set default values based on fiber mode (most common)
    return render(request, 'cowqkd.html', {
        'channel_mode': 'fiber',
        'default_distance': 50,
        'default_mu': 0.3,
        'default_distance_max_qber': 200,
        'default_distance_max_skr': 200
    })


def dps_modified_plot_qber_vs_mu(simulator, mu_range, points=100):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values
    """
    mu_values, qber_values = simulator.get_qber_vs_mu_data(mu_range, points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2, 
             label=f'QBER ({simulator.channel_mode} channel)')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('QBER', fontsize=12)
    plt.title(f'QBER vs Mean Photon Number - {simulator.channel_mode.upper()} Channel', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, mu_values, qber_values

def dps_modified_plot_qber_vs_distance(simulator, distance_range, points=100):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot
    """
    distance_values, qber_values = simulator.get_qber_vs_distance_data(distance_range, points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'go-', linewidth=2, 
             label=f'QBER ({simulator.channel_mode} channel)')
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('QBER', fontsize=12)
    plt.title(f'QBER vs Distance - {simulator.channel_mode.upper()} Channel', fontsize=14)
    plt.legend(fontsize=10)
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
    plt.plot(mu_values, skr_values, 'ro-', linewidth=2, 
             label=f'SKR ({simulator.channel_mode} channel)')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('SKR (bits/s)', fontsize=12)
    plt.title(f'Secret Key Rate vs Mean Photon Number - {simulator.channel_mode.upper()} Channel', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, mu_values, skr_values

def dps_modified_plot_skr_vs_distance(simulator, distance_range, points=100):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot
    """
    distance_values, skr_values = simulator.get_skr_vs_distance_data(distance_range, points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'mo-', linewidth=2, 
             label=f'SKR ({simulator.channel_mode} channel)')
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('SKR (bits/s)', fontsize=12)
    plt.title(f'Secret Key Rate vs Distance - {simulator.channel_mode.upper()} Channel', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, distance_values, skr_values

def dpsqkd(request):
    """
    View function to handle the DPS QKD simulator form and run simulations.
    """
    # Default parameters for both channel modes
    default_params = {
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
        'distance_max': 200,
        'channel_mode': 'fiber',
        # FSO specific parameters
        'transmitter_diameter': 0.1,
        'receiver_diameter': 0.3,
        'beam_divergence': 0.001,
        'wavelength': 850e-9,
        'pointing_error': 1e-6,
        'transmitter_efficiency': 0.9,
        'receiver_efficiency': 0.9
    }
    
    if request.method == 'POST':
        # Get form data
        nem = int(float(request.POST.get('nem', default_params['nem'])))
        repetition_rate = float(request.POST.get('repetition_rate', default_params['repetition_rate']))
        mu = float(request.POST.get('mu', default_params['mu']))
        ebit = float(request.POST.get('ebit', default_params['ebit']))
        eta_det = float(request.POST.get('eta_det', default_params['eta_det']))
        dark_count = float(request.POST.get('dark_count', default_params['dark_count']))
        alpha = float(request.POST.get('alpha', default_params['alpha']))
        distance = float(request.POST.get('distance', default_params['distance']))
        delta_bs1 = float(request.POST.get('delta_bs1', default_params['delta_bs1']))
        delta_bs2 = float(request.POST.get('delta_bs2', default_params['delta_bs2']))
        t_prob = float(request.POST.get('t_prob', default_params['t_prob']))
        channel_mode = request.POST.get('channel_mode', default_params['channel_mode'])
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', default_params['mu_min']))
        mu_max = float(request.POST.get('mu_max', default_params['mu_max']))
        distance_max = float(request.POST.get('distance_max', default_params['distance_max']))
        
        # FSO specific parameters (only used if channel_mode is "fso")
        transmitter_diameter = float(request.POST.get('transmitter_diameter', default_params['transmitter_diameter']))
        receiver_diameter = float(request.POST.get('receiver_diameter', default_params['receiver_diameter']))
        beam_divergence = float(request.POST.get('beam_divergence', default_params['beam_divergence']))
        wavelength = float(request.POST.get('wavelength', default_params['wavelength']))
        pointing_error = float(request.POST.get('pointing_error', default_params['pointing_error']))
        transmitter_efficiency = float(request.POST.get('transmitter_efficiency', default_params['transmitter_efficiency']))
        receiver_efficiency = float(request.POST.get('receiver_efficiency', default_params['receiver_efficiency']))
        
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
            t_prob=t_prob,
            channel_mode=channel_mode,
            transmitter_diameter=transmitter_diameter,
            receiver_diameter=receiver_diameter,
            beam_divergence=beam_divergence,
            wavelength=wavelength,
            pointing_error=pointing_error,
            transmitter_efficiency=transmitter_efficiency,
            receiver_efficiency=receiver_efficiency
        )
        
        # Generate plots
        qber_vs_mu_plot, mu_values, qber_values = dps_modified_plot_qber_vs_mu(
            simulator, mu_range, points=100
        )
        
        skr_vs_mu_plot, _, skr_mu_values = dps_modified_plot_skr_vs_mu(
            simulator, mu_range, points=100
        )
        
        # For FSO mode, limit distance range to a more reasonable value
        if channel_mode == 'fso':
            distance_range = (0, min(25, distance_max))
            
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
        
        context = {
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
            'channel_mode': channel_mode,
            'optimal_mu': f"{optimal_mu:.4f}",
            'max_distance': f"{max_distance:.1f}",
            'current_qber': f"{current_qber:.6f}",
            'current_skr': f"{current_skr:.2f}",
            'mu_min': mu_min,
            'mu_max': mu_max,
            'distance_max': distance_max,
        }
        
        # Add FSO parameters to context if FSO mode is selected
        if channel_mode == 'fso':
            fso_params = {
                'transmitter_diameter': transmitter_diameter,
                'receiver_diameter': receiver_diameter,
                'beam_divergence': beam_divergence,
                'wavelength': wavelength * 1e9,  # Convert to nm for display
                'pointing_error': pointing_error,
                'transmitter_efficiency': transmitter_efficiency,
                'receiver_efficiency': receiver_efficiency,
            }
            context.update(fso_params)
        
        return render(request, 'dpsqkd.html', context)
    
    # If GET request, just render the form with default values
    return render(request, 'dpsqkd.html', default_params)


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

def bb84_plot_qber_vs_mu(mu_values, time_window, distance,
                          detector_efficiency, channel_base_efficiency, 
                          dark_count_rate, attenuation=0.2, p_eve=0.0,
                          channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values. Supports both fiber and FSO channels.
    """
    qber_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            attenuation=attenuation,
            p_eve=p_eve,
            channel_mode=channel_mode
        )
        
        # Set custom FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=11, color='red', linestyle='--', label='11% threshold')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('QBER (%)')
    plt.title(f'Quantum Bit Error Rate vs Mean Photon Number ({channel_mode.upper()}, {distance} km)')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values


def bb84_plot_skr_vs_mu(mu_values, time_window, distance,
                         detector_efficiency, channel_base_efficiency, 
                         dark_count_rate, repetition_rate, attenuation=0.2, p_eve=0.0,
                         channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_skr_vs_mu that returns the base64 encoded plot
    and the SKR values. Supports both fiber and FSO channels.
    """
    skr_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            attenuation=attenuation,
            p_eve=p_eve,
            channel_mode=channel_mode
        )
        
        # Set custom FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title(f'Secret Key Rate vs Mean Photon Number ({channel_mode.upper()}, {distance} km)')
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values


def bb84_plot_qber_vs_distance(distance_values, time_window, mu,
                               detector_efficiency, channel_base_efficiency, 
                               dark_count_rate, attenuation=0.2, p_eve=0.0,
                               channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot.
    Supports both fiber and FSO channels.
    """
    qber_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        detector_efficiency=detector_efficiency,
        channel_base_efficiency=channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        attenuation=attenuation,
        p_eve=p_eve,
        channel_mode=channel_mode
    )
    
    # Set custom FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'ro-', linewidth=2)
    plt.grid(True)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=11, color='red', linestyle='--', label='11% threshold')
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title(f'Quantum Bit Error Rate vs Distance ({channel_mode.upper()}, μ={mu})')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values


def bb84_plot_skr_vs_distance(distance_values, time_window, mu,
                              detector_efficiency, channel_base_efficiency, 
                              dark_count_rate, repetition_rate, attenuation=0.2, p_eve=0.0,
                              channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot.
    Supports both fiber and FSO channels.
    """
    skr_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        detector_efficiency=detector_efficiency,
        channel_base_efficiency=channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        attenuation=attenuation,
        p_eve=p_eve,
        channel_mode=channel_mode
    )
    
    # Set custom FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
    for distance in distance_values:
        simulator.update_distance(distance)
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(distance_values, skr_values, 'mo-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title(f'Secret Key Rate vs Distance ({channel_mode.upper()}, μ={mu})')
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values


def bb84_plot_qber_skr_vs_eavesdropping(p_eve_values, time_window, distance, mu,
                                         detector_efficiency, channel_base_efficiency, 
                                         dark_count_rate, repetition_rate, attenuation=0.2,
                                         channel_mode="fiber", fso_params=None):
    """
    Modified version of plot_qber_skr_vs_eavesdropping that returns the base64 encoded plots.
    Supports both fiber and FSO channels.
    """
    qber_values = []
    skr_values = []
    
    for p_eve in p_eve_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            attenuation=attenuation,
            p_eve=p_eve,
            channel_mode=channel_mode
        )
        
        # Set custom FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
        
        # Calculate QBER
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
        
        # Calculate SKR
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    # Plot QBER vs p_eve
    plt.figure(figsize=(10, 6))
    plt.plot(p_eve_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=11, color='red', linestyle='--', label='Security Threshold (11%)')
    plt.grid(True)
    plt.xlabel('Eavesdropping Probability (p_eve)')
    plt.ylabel('QBER (%)')
    plt.title(f'Quantum Bit Error Rate vs Eavesdropping Probability ({channel_mode.upper()}, {distance} km)')
    plt.legend()
    
    qber_plot = get_plot_base64(plt)
    
    # Plot SKR vs p_eve
    plt.figure(figsize=(10, 6))
    plt.plot(p_eve_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Eavesdropping Probability (p_eve)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title(f'Secret Key Rate vs Eavesdropping Probability ({channel_mode.upper()}, {distance} km)')
    
    skr_plot = get_plot_base64(plt)
    
    return qber_plot, skr_plot, qber_values, skr_values

def bb84(request):
    """
    View function to handle the BB84 simulator form and run simulations.
    Supports both fiber and FSO channel modes.
    """
    # Default values for parameters - fiber mode
    fiber_defaults = {
        'mu': 0.1,
        'distance': 50,
        'attenuation': 0.2,
        'detector_efficiency': 0.15,
        'channel_base_efficiency': 0.6,
        'dark_count_rate': 2000,
        'time_window': 10,  # in ns
        'repetition_rate': 1000000,
        'p_eve': 0.0,
        'mu_min': 0.01,
        'mu_max': 2.0,
        'distance_max_qber': 120,
        'distance_max_skr': 120,
        'p_eve_max': 0.5,
        'channel_mode': 'fiber',
    }
    
    # FSO mode default parameters based on example code
    fso_defaults = {
        'channel_mode': 'fso',
        'distance': 1,                  # 1 km default for FSO
        'distance_max_qber': 20,        # 20 km max for QBER vs distance plots
        'distance_max_skr': 5,          # 5 km max for SKR vs distance plots
        'transmitter_diameter': 0.1,    # 10 cm
        'receiver_diameter': 0.3,       # 30 cm
        'beam_divergence': 1,           # 1 mrad (will be converted to radians)
        'wavelength': 850,              # 850 nm (will be converted to meters)
        'pointing_error': 1             # 1 μrad (will be converted to radians)
    }
    
    if request.method == 'POST':
        # Get channel mode first to determine appropriate defaults
        channel_mode = request.POST.get('channel_mode', fiber_defaults['channel_mode'])
        
        # Use appropriate defaults based on channel mode
        current_defaults = fiber_defaults.copy()
        if channel_mode == "fso":
            current_defaults.update(fso_defaults)
        
        # Get basic parameters with defaults if missing
        mu = float(request.POST.get('mu', current_defaults['mu']))
        distance = float(request.POST.get('distance', current_defaults['distance']))
        attenuation = float(request.POST.get('attenuation', current_defaults['attenuation']))
        detector_efficiency = float(request.POST.get('detector_efficiency', current_defaults['detector_efficiency']))
        channel_base_efficiency = float(request.POST.get('channel_base_efficiency', current_defaults['channel_base_efficiency']))
        dark_count_rate = float(request.POST.get('dark_count_rate', current_defaults['dark_count_rate']))
        time_window = float(request.POST.get('time_window', current_defaults['time_window'])) * 1e-9  # Convert ns to seconds
        repetition_rate = float(request.POST.get('repetition_rate', current_defaults['repetition_rate']))
        p_eve = float(request.POST.get('p_eve', current_defaults['p_eve']))
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', current_defaults['mu_min']))
        mu_max = float(request.POST.get('mu_max', current_defaults['mu_max']))
        distance_max_qber = float(request.POST.get('distance_max_qber', current_defaults['distance_max_qber']))
        distance_max_skr = float(request.POST.get('distance_max_skr', current_defaults['distance_max_skr']))
        p_eve_max = float(request.POST.get('p_eve_max', current_defaults['p_eve_max']))
        
        # Get FSO specific parameters and convert to correct units
        if channel_mode == "fso":
            transmitter_diameter = float(request.POST.get('transmitter_diameter', current_defaults['transmitter_diameter']))
            receiver_diameter = float(request.POST.get('receiver_diameter', current_defaults['receiver_diameter']))
            beam_divergence = float(request.POST.get('beam_divergence', current_defaults['beam_divergence'])) * 1e-3  # Convert mrad to rad
            wavelength = float(request.POST.get('wavelength', current_defaults['wavelength'])) * 1e-9  # Convert nm to m
            pointing_error = float(request.POST.get('pointing_error', current_defaults['pointing_error'])) * 1e-6  # Convert μrad to rad
            
            # Set FSO parameters dictionary with proper units for simulator
            fso_params = {
                'transmitter_diameter': transmitter_diameter,
                'receiver_diameter': receiver_diameter,
                'beam_divergence': beam_divergence,  # Already converted to radians
                'wavelength': wavelength,           # Already converted to meters
                'pointing_error': pointing_error    # Already converted to radians
            }
        else:
            # Set empty defaults for display when in fiber mode
            transmitter_diameter = 0.1
            receiver_diameter = 0.3
            beam_divergence = 1
            wavelength = 850
            pointing_error = 1
            fso_params = None
        
        # Define plot ranges - use starting point of 0.1 km to avoid division by zero in FSO mode
        mu_values_qber = np.linspace(mu_min, mu_max, 40)
        mu_values_skr = np.linspace(mu_min, min(mu_max, 1.2), 40)
        distance_start = 0.1 if channel_mode == "fso" else 0  # Start at 0.1 km for FSO to avoid division by zero
        distance_values_qber = np.linspace(distance_start, distance_max_qber, 40)
        distance_values_skr = np.linspace(distance_start, distance_max_skr, 40)
        p_eve_values = np.linspace(0, p_eve_max, 50)
        
        # Use the appropriate μ values based on examples
        qber_mu = 0.1  # Default mu for QBER vs distance
        skr_mu = 0.1   # Default mu for SKR vs distance and eavesdropping
        
        # Generate QBER vs mu plot
        qber_vs_mu_plot, qber_values = bb84_plot_qber_vs_mu(
            mu_values_qber, time_window, distance,
            detector_efficiency, channel_base_efficiency, 
            dark_count_rate, attenuation, p_eve,
            channel_mode, fso_params
        )
        
        # Generate SKR vs mu plot
        skr_vs_mu_plot, skr_values = bb84_plot_skr_vs_mu(
            mu_values_skr, time_window, distance,
            detector_efficiency, channel_base_efficiency, 
            dark_count_rate, repetition_rate, attenuation, p_eve,
            channel_mode, fso_params
        )
        
        # Find optimal μ value for best SKR
        if len(skr_values) > 0:
            optimal_mu_index = np.argmax(skr_values)
            optimal_mu = mu_values_skr[optimal_mu_index]
        else:
            optimal_mu = mu  # Use current mu if no values calculated
        
        # Generate QBER vs distance plot
        qber_vs_distance_plot, _ = bb84_plot_qber_vs_distance(
            distance_values_qber, time_window, qber_mu,
            detector_efficiency, channel_base_efficiency, 
            dark_count_rate, attenuation, p_eve,
            channel_mode, fso_params
        )
        
        # Generate SKR vs distance plot
        skr_vs_distance_plot, _ = bb84_plot_skr_vs_distance(
            distance_values_skr, time_window, skr_mu,
            detector_efficiency, channel_base_efficiency, 
            dark_count_rate, repetition_rate, attenuation, p_eve,
            channel_mode, fso_params
        )
        
        # Generate eavesdropping plots
        qber_vs_eve_plot, skr_vs_eve_plot, _, _ = bb84_plot_qber_skr_vs_eavesdropping(
            p_eve_values, time_window, distance, skr_mu,
            detector_efficiency, channel_base_efficiency, 
            dark_count_rate, repetition_rate, attenuation,
            channel_mode, fso_params
        )
        
        # Calculate additional metrics with optimal μ
        simulator = BB84Simulator(
            mu=optimal_mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            attenuation=attenuation,
            p_eve=p_eve,
            channel_mode=channel_mode
        )
        
        # Set FSO parameters if needed
        if channel_mode == "fso":
            simulator.set_fso_parameters(**fso_params)
        
        optimal_qber = simulator.calculate_quantum_bit_error_rate()
        optimal_skr = simulator.calculate_skr() * repetition_rate
        
        # Find maximum secure distance (where QBER ≤ 11%)
        max_distance = 0
        # Use smaller step for FSO and limit search range based on mode
        distance_step = 0.1 if channel_mode == "fso" else 5
        max_search_distance = min(distance_max_qber, 20 if channel_mode == "fso" else 120)
        
        for d in np.arange(distance_start, max_search_distance, distance_step):
            simulator.update_distance(d)
            qber = simulator.calculate_quantum_bit_error_rate()
            if qber <= 11:
                max_distance = d
            else:
                break
        
        # Calculate photon number distribution
        photon_source = WeakCoherentSource(optimal_mu)
        photon_dist = photon_source.photon_distribution(10)  # Distribution up to 10 photons
        
        # Convert distribution to percentage for display
        photon_percentages = [p * 100 for p in photon_dist[:5]]  # Show first 5 values
        
        # Package results for the template
        plots = {
            'qber_vs_mu': qber_vs_mu_plot,
            'skr_vs_mu': skr_vs_mu_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'skr_vs_distance': skr_vs_distance_plot,
            'qber_vs_eve': qber_vs_eve_plot,
            'skr_vs_eve': skr_vs_eve_plot
        }
        
        # Get current channel efficiency
        if channel_mode == "fiber":
            current_channel = Channel(channel_base_efficiency, distance, attenuation)
        else:
            current_channel = Channel(channel_base_efficiency, distance, attenuation, mode="fso")
            current_channel.set_fso_parameters(**fso_params)
            
        channel_efficiency = current_channel.efficiency * 100  # as percentage
        
        # Format parameters for display in the template - convert units back for display
        form_data = {
            'mu': mu,
            'distance': distance,
            'attenuation': attenuation,
            'detector_efficiency': detector_efficiency,
            'channel_base_efficiency': channel_base_efficiency,
            'dark_count_rate': dark_count_rate,
            'time_window': time_window * 1e9,  # Convert back to ns for display
            'repetition_rate': repetition_rate,
            'p_eve': p_eve,
            'mu_min': mu_min,
            'mu_max': mu_max,
            'distance_max_qber': distance_max_qber,
            'distance_max_skr': distance_max_skr,
            'p_eve_max': p_eve_max,
            'channel_mode': channel_mode,
            'transmitter_diameter': transmitter_diameter,
            'receiver_diameter': receiver_diameter,
            'beam_divergence': beam_divergence * 1000 if channel_mode == "fso" else beam_divergence,  # Convert rad to mrad for display
            'wavelength': wavelength * 1e9 if channel_mode == "fso" else wavelength,  # Convert m to nm for display
            'pointing_error': pointing_error * 1e6 if channel_mode == "fso" else pointing_error  # Convert rad to μrad for display
        }
        
        return render(request, 'bb84.html', {
            'plots': plots,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_qber': f"{optimal_qber:.4f}",
            'optimal_skr': f"{optimal_skr:.2f}",
            'max_distance': f"{max_distance:.1f}",
            'channel_efficiency': f"{channel_efficiency:.4f}",
            'photon_percentages': [f"{p:.2f}" for p in photon_percentages],
            'form_data': form_data,
            'submission': True
        })
    else:
        # For GET request, render the form with default values
        channel_mode = request.GET.get('channel_mode', fiber_defaults['channel_mode'])
        
        # Use appropriate defaults based on channel mode
        current_defaults = fiber_defaults.copy()
        if channel_mode == "fso":
            current_defaults.update(fso_defaults)
        
        # Convert time_window from ns to seconds for simulator
        time_window = current_defaults['time_window'] * 1e-9
        
        # Prepare FSO parameters with correct units if needed
        fso_params = None
        if channel_mode == "fso":
            fso_params = {
                'transmitter_diameter': current_defaults['transmitter_diameter'],
                'receiver_diameter': current_defaults['receiver_diameter'],
                'beam_divergence': current_defaults['beam_divergence'] * 1e-3,  # Convert mrad to rad
                'wavelength': current_defaults['wavelength'] * 1e-9,  # Convert nm to m
                'pointing_error': current_defaults['pointing_error'] * 1e-6  # Convert μrad to rad
            }
        
        # Define plot ranges based on mode
        mu_values_qber = np.linspace(current_defaults['mu_min'], current_defaults['mu_max'], 40)
        mu_values_skr = np.linspace(current_defaults['mu_min'], min(current_defaults['mu_max'], 1.2), 40)
        distance_start = 0.1 if channel_mode == "fso" else 0
        distance_values_qber = np.linspace(distance_start, current_defaults['distance_max_qber'], 40)
        distance_values_skr = np.linspace(distance_start, current_defaults['distance_max_skr'], 40)
        p_eve_values = np.linspace(0, current_defaults['p_eve_max'], 50)
        
        # Use appropriate mu values based on examples
        qber_mu = 0.1  # Default mu for QBER vs distance
        skr_mu = 0.1   # Default mu for SKR vs distance and eavesdropping
        
        # Generate default plots
        qber_vs_mu_plot, _ = bb84_plot_qber_vs_mu(
            mu_values_qber, time_window, current_defaults['distance'],
            current_defaults['detector_efficiency'], current_defaults['channel_base_efficiency'], 
            current_defaults['dark_count_rate'], current_defaults['attenuation'], current_defaults['p_eve'],
            channel_mode, fso_params
        )
        
        skr_vs_mu_plot, _ = bb84_plot_skr_vs_mu(
            mu_values_skr, time_window, current_defaults['distance'],
            current_defaults['detector_efficiency'], current_defaults['channel_base_efficiency'], 
            current_defaults['dark_count_rate'], current_defaults['repetition_rate'], 
            current_defaults['attenuation'], current_defaults['p_eve'],
            channel_mode, fso_params
        )
        
        qber_vs_distance_plot, _ = bb84_plot_qber_vs_distance(
            distance_values_qber, time_window, qber_mu,
            current_defaults['detector_efficiency'], current_defaults['channel_base_efficiency'], 
            current_defaults['dark_count_rate'], current_defaults['attenuation'], current_defaults['p_eve'],
            channel_mode, fso_params
        )
        
        skr_vs_distance_plot, _ = bb84_plot_skr_vs_distance(
            distance_values_skr, time_window, skr_mu,
            current_defaults['detector_efficiency'], current_defaults['channel_base_efficiency'], 
            current_defaults['dark_count_rate'], current_defaults['repetition_rate'], 
            current_defaults['attenuation'], current_defaults['p_eve'],
            channel_mode, fso_params
        )
        
        qber_vs_eve_plot, skr_vs_eve_plot, _, _ = bb84_plot_qber_skr_vs_eavesdropping(
            p_eve_values, time_window, current_defaults['distance'], skr_mu,
            current_defaults['detector_efficiency'], current_defaults['channel_base_efficiency'], 
            current_defaults['dark_count_rate'], current_defaults['repetition_rate'], 
            current_defaults['attenuation'], channel_mode, fso_params
        )
        
        # Package default plots for the template
        plots = {
            'qber_vs_mu': qber_vs_mu_plot,
            'skr_vs_mu': skr_vs_mu_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'skr_vs_distance': skr_vs_distance_plot,
            'qber_vs_eve': qber_vs_eve_plot,
            'skr_vs_eve': skr_vs_eve_plot
        }
        
        # Prepare display values for the template
        display_defaults = current_defaults.copy()
        
        # Display FSO parameters in their proper display units
        if channel_mode == "fso":
            # No need to convert these as they're already in display units in defaults
            pass
            
        return render(request, 'bb84.html', {
            'plots': plots,
            'form_data': display_defaults,
            'submission': False
        })