import numpy as np
import matplotlib.pyplot as plt


class MuseumSimulation:
    def __init__(self,
                 n_visitors=200,  # Total visitors for the day
                 n_exhibitions=30,  # Number of exhibitions
                 total_time=480,  # Total simulation time (minutes)
                 morning_peak=120,  # Morning peak time (minutes after start)
                 afternoon_peak=360):  # Afternoon peak time (minutes after start)

        self.N = n_visitors
        self.n_exhibitions = n_exhibitions
        self.total_time = total_time
        self.morning_peak = morning_peak
        self.afternoon_peak = afternoon_peak

        # Parameters for entry distribution
        self.morning_std = 30
        self.afternoon_std = 40

        # Museum parameters
        self.T = 1  # Temperature (randomness)
        self.dt = 0.1  # Time step
        self.k = 1  # Well strength
        self.drift = 3 # Drift

        # Exhibition positions
        self.well_positions = np.linspace(-n_exhibitions + 1, n_exhibitions - 1, n_exhibitions)
        self.well_depths = np.ones(n_exhibitions)
        self.left_boundary = self.well_positions[0] - 1  # Define the left boundary

        # Tracking visitors
        self.visitor_data = []

    def entry_probability(self, t):
        """Calculate visitor entry probability at time t"""
        morning_prob = np.exp(-((t - self.morning_peak) ** 2) / (2 * self.morning_std ** 2))
        afternoon_prob = np.exp(-((t - self.afternoon_peak) ** 2) / (2 * self.afternoon_std ** 2))
        return morning_prob + afternoon_prob

    def potential(self, x):
        """Calculate multi-well potential"""
        V = np.zeros_like(x)
        for pos, depth in zip(self.well_positions, self.well_depths):
            V += -depth * self.k * np.exp(-(x - pos) ** 2)
        return V

    def force(self, x, current_time):
        """Calculate force with time-based drift toward the exit."""
        eps = 1e-6
        # Standard force from potential
        base_force = -(self.potential(x + eps) - self.potential(x - eps)) / (2 * eps)

        # Time-based drift toward the right boundary (exit)
        exit_bias = self.drift * (current_time / self.total_time)  # Drift increases over time
        exit_force = exit_bias if x < self.well_positions[-1] else 0  # Apply only before the exit

        return base_force + exit_force

    def generate_noise(self, size):
        """Generate thermal noise"""
        return np.sqrt(2 * self.T * self.dt) * np.random.normal(size=size)

    def simulate(self):
        time = np.arange(0, self.total_time, self.dt)

        # Preallocate memory for tracking
        max_visitors = int(self.N * 1.5)  # Allow some buffer
        positions = np.zeros((max_visitors, len(time))) + np.nan

        active_visitors = []

        for i, t in enumerate(time):
            # Add new visitors based on time-dependent probability
            new_visitors = np.random.random() < self.entry_probability(t) * self.dt
            if new_visitors and len(active_visitors) < max_visitors:
                # Add new visitor at museum entrance
                new_visitor_id = len(self.visitor_data)
                new_pos = np.random.normal(self.well_positions[0], 0.2)

                visitor_info = {
                    'id': new_visitor_id,
                    'entry_time': t,
                    'exit_time': None,
                    'exhibition_times': np.zeros(self.n_exhibitions),
                    'positions': []
                }

                active_visitors.append(new_visitor_id)
                self.visitor_data.append(visitor_info)
                positions[new_visitor_id, i] = new_pos

            # Update positions of active visitors
            for j, visitor_id in enumerate(active_visitors):
                visitor = self.visitor_data[visitor_id]

                if np.isnan(positions[visitor_id, i - 1]) and i > 0:
                    continue

                # Initial position or previous position
                curr_pos = positions[visitor_id, i - 1] if i > 0 else self.well_positions[0]

                # Langevin dynamics
                noise = self.generate_noise(1)
                new_pos = curr_pos + self.force(curr_pos, t) * self.dt + noise

                # Prevent visitor from exiting the left boundary
                if new_pos < self.left_boundary:
                    new_pos = np.array([self.left_boundary])

                positions[visitor_id, i] = new_pos
                visitor['positions'].append(new_pos)

                # Track time spent at each exhibition
                for k, well_pos in enumerate(self.well_positions):
                    if abs(new_pos - well_pos) < 0.5:
                        visitor['exhibition_times'][k] += self.dt

                # Check for exit
                if new_pos > self.well_positions[-1]:
                    visitor['exit_time'] = t
                    active_visitors.pop(j)
                    break

        return {
            'time': time,
            'positions': positions,
            'visitor_data': self.visitor_data
        }

    def analyze_results(self, results):
        """Calculate statistics for completed visits"""
        # Filter completed visits
        completed_visitors = [v for v in results['visitor_data'] if v['exit_time'] is not None]

        # Cost statistics
        total_cost = 0
        for visitor in completed_visitors:
            visitor_cost = 0
            for exhibition_time in visitor['exhibition_times']:
                visitor_cost += ((0.06 * 0.25) + (0.24 * 0.75)) * exhibition_time + \
                                ((exhibition_time * (exhibition_time - 1.0) / 2.0) * 150.0 + 2400.0) * 0.000005
            total_cost += visitor_cost

        avg_cost_per_person = total_cost / len(completed_visitors)

        # Calculate statistics
        stats = {
            'total_visitors': len(results['visitor_data']),
            'completed_visits': len(completed_visitors),
            'visit_durations': [v['exit_time'] - v['entry_time'] for v in completed_visitors],
            'exhibition_times': [v['exhibition_times'] for v in completed_visitors],
            'total_cost': total_cost,
            'avg_cost_per_person': avg_cost_per_person
        }

        # Compute summary statistics
        stats['mean_visit_duration'] = np.mean(stats['visit_durations']) if stats['visit_durations'] else 0
        stats['mean_exhibition_times'] = np.mean(stats['exhibition_times'], axis=0) if stats[
            'exhibition_times'] else None

        return stats

    def plot_potential_landscape(self, results):
        """Plot the potential landscape and histogram of particle positions"""
        x = np.linspace(-self.n_exhibitions + 1, self.n_exhibitions - 1, 200)
        V = self.potential(x)

        positions = results['positions'][~np.isnan(results['positions'])]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot potential landscape
        ax1.plot(x, V, color='blue')
        ax1.set_title('Museum Potential Landscape')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Potential Energy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Mark exhibition positions
        for pos in self.well_positions:
            ax1.axvline(x=pos, color='r', linestyle='--', alpha=0.3)

        # Create a secondary y-axis for the histogram
        ax2 = ax1.twinx()
        ax2.hist(positions, bins=60, density=True, alpha=0.6, color='green')
        ax2.set_ylabel('Frequency', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.show()

    def plot_results(self, results):
        """Visualize simulation results"""
        # Filter out visitors who completed their visit
        completed_visitors = [v for v in results['visitor_data'] if v['exit_time'] is not None]

        plt.figure(figsize=(15, 10))

        # Visitor paths
        plt.subplot(2, 2, 1)
        for visitor in completed_visitors[:10]:
            plt.plot(results['time'][:len(visitor['positions'])], visitor['positions'])
        plt.title('Sample Visitor Paths')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Position')

        # Exhibition times
        plt.subplot(2, 2, 2)
        exhibition_times = np.array([v['exhibition_times'] for v in completed_visitors])
        plt.boxplot(exhibition_times)
        plt.title('Time Spent at Each Exhibition')
        plt.xlabel('Exhibition Number')
        plt.ylabel('Time (minutes)')

        # Visit duration distribution
        plt.subplot(2, 2, 3)
        visit_durations = np.array([v['exit_time'] - v['entry_time'] for v in completed_visitors])
        plt.hist(visit_durations, bins=25)
        plt.title('Distribution of Visit Durations')
        plt.xlabel('Visit Duration (minutes)')
        plt.ylabel('Number of Visitors')

        # Entry times distribution
        plt.subplot(2, 2, 4)
        entry_times = np.array([v['entry_time'] for v in completed_visitors])
        plt.hist(entry_times, bins=25)
        plt.title('Visitor Entry Distribution')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Number of Visitors')

        plt.tight_layout()
        plt.show()


# Run simulation
sim = MuseumSimulation(n_visitors=300, n_exhibitions=30)
results = sim.simulate()

# Plot potential landscape
sim.plot_potential_landscape(results)

# Analyze and print statistics
stats = sim.analyze_results(results)
print("\nSimulation Statistics:")
print(f"Total visitors: {stats['total_visitors']}")
print(f"Completed visits: {stats['completed_visits']}")
print(f"Mean visit duration: {stats['mean_visit_duration']:.1f} minutes")
print("Mean time at each exhibition:", stats['mean_exhibition_times'])


# cost statistics
print(f"Total cost: {stats['total_cost']:.2f}")
print(f"Average cost per person: {stats['avg_cost_per_person']:.2f}")

# Visualize results
sim.plot_results(results)