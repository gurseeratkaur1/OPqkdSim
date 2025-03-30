from django.shortcuts import render
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ThOPqkdsim.sim6 import *
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
                     dark_count_rate):
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
        skr = simulator.calculate_skr(key_length)
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('Secret Key Rate (bits per channel use)')
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
                          dark_count_rate):
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
        skr = simulator.calculate_skr(key_length)
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'mo-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secret Key Rate (bits per channel use)')
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
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', 0.01))
        mu_max = float(request.POST.get('mu_max', 1.0))
        distance_max_qber = float(request.POST.get('distance_max_qber', 300))
        distance_max_skr = float(request.POST.get('distance_max_skr', 210))
        
        # Define plot ranges
        mu_values = np.linspace(mu_min, mu_max, 20)
        distance_values_qber = np.linspace(0, distance_max_qber, 50)
        distance_values_skr = np.linspace(0, distance_max_skr, 50)
        
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
            dark_count_rate
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
            dark_count_rate
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
        for d in np.arange(0, 500, 1):
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
        
        return render(request, 'bb84.html', {
            'plots': plots,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_qber': f"{optimal_qber:.4f}",
            'optimal_skr': f"{optimal_skr:.6f}",
            'max_distance': f"{max_distance:.1f}"
        })
    
    # If GET request, just render the form
    return render(request, 'bb84.html')