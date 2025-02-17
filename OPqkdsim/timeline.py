import time
import heapq
import numpy as np

class Timeline:
    """
    Real-time Timeline class that manages and executes scheduled events.
    It ensures simulation timing aligns with real-world execution time.
    """

    def __init__(self, dt=1e-12):
        """
        Initializes the timeline.
        
        Args:
            dt (float): Default time step for calculations (seconds).
        """
        self.t_start = time.time()  # Capture real-world start time
        self.current_time = self.t_start  # Initialize current time
        self.dt = dt
        self.event_queue = []  # Min-heap (priority queue) for events
        self.subscribers = {}  # Store component event handlers

    def schedule_event(self, delay, event_function, *args, **kwargs):
        """
        Schedules an event with a real-time delay.

        Args:
            delay (float): Delay in seconds from the current time.
            event_function (callable): Function to execute.
            *args, **kwargs: Additional parameters for the function.
        """
        event_time = time.time() + delay
        heapq.heappush(self.event_queue, (event_time, event_function, args, kwargs))

    def execute_events(self, max_events=None):
        """
        Executes scheduled events in real-time.

        Args:
            max_events (int, optional): Maximum number of events to process. 
                                        If None, runs until the queue is empty.
        """
        event_count = 0

        while self.event_queue:
            if max_events and event_count >= max_events:
                break

            event_time, event_function, args, kwargs = heapq.heappop(self.event_queue)
            time_to_wait = max(0, event_time - time.time())  # Avoid negative delay

            if time_to_wait > 0:
                time.sleep(time_to_wait)  # Real-time delay

            self.current_time = time.time()  # Update current time
            event_function(self.current_time, *args, **kwargs)  # Execute event
            event_count += 1

    def get_current_time(self):
        """Returns the current real-world simulation time."""
        return time.time()

    def get_elapsed_time(self):
        """Returns elapsed simulation time since start."""
        return time.time() - self.t_start

    def publish(self, sender, *args, **kwargs):
        """Forwards signals to the next subscribed component."""
        if sender in self.subscribers:
            handler = self.subscribers[sender]
            self.schedule_event(self.dt, handler, *args, **kwargs)

    def subscribe(self, component, handler):
        """Registers a component to receive signals."""
        self.subscribers[component] = handler