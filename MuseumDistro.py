import numpy as np
import matplotlib.pyplot as plt

class MarkovMuseumSimulation:
    def __init__(self, n_visitors=300, n_exhibitions=30, total_time=480, morning_peak=120, afternoon_peak=360, transition_prob=0.6):
        self.n_visitors = n_visitors
        self.n_exhibitions = n_exhibitions
        self.total_time = total_time
        self.morning_peak = morning_peak
        self.afternoon_peak = afternoon_peak
        self.transition_prob = transition_prob

        self.morning_std = 30
        self.afternoon_std = 40

        self.dt = 0.1  # Time step
        self.visitor_data = []

        # Transition probabilities for each exhibition
        self.exhibition_transition_probs = np.full(n_exhibitions - 1, self.transition_prob)

    def entry_probability(self, t):
        """Calculate visitor entry probability at time t."""
        morning_prob = np.exp(-((t - self.morning_peak) ** 2) / (2 * self.morning_std ** 2))
        afternoon_prob = np.exp(-((t - self.afternoon_peak) ** 2) / (2 * self.afternoon_std ** 2))
        return morning_prob + afternoon_prob

    def simulate(self):
        time = np.arange(0, self.total_time, self.dt)
        visitors_in_museum = []
        visitor_data = []

        # Transition matrix for exhibitions
        transition_matrix = np.zeros((self.n_exhibitions, self.n_exhibitions))
        for i in range(self.n_exhibitions - 1):
            transition_matrix[i, i] = 1 - self.exhibition_transition_probs[i]
            transition_matrix[i, i + 1] = self.exhibition_transition_probs[i]
        transition_matrix[-1, -1] = 1.0  # Absorbing state

        total_visitors = 0

        for t in time:
            # New visitors entry
            new_visitors = np.random.random() < self.entry_probability(t) * self.dt
            if new_visitors and total_visitors < self.n_visitors:
                visitor = {
                    'id': total_visitors,
                    'entry_time': t,
                    'current_exhibition': 0,
                    'exhibition_times': np.zeros(self.n_exhibitions),
                    'exit_time': None
                }
                visitors_in_museum.append(visitor)
                visitor_data.append(visitor)
                total_visitors += 1

            # Update positions of visitors
            for visitor in visitors_in_museum[:]:
                current_exhibition = visitor['current_exhibition']
                visitor['exhibition_times'][current_exhibition] += self.dt

                # Determine transition to the next exhibition
                if current_exhibition < self.n_exhibitions - 1:
                    if np.random.random() < transition_matrix[current_exhibition, current_exhibition + 1]:
                        visitor['current_exhibition'] += 1

                # Check if visitor has exited
                if visitor['current_exhibition'] == self.n_exhibitions - 1:
                    visitor['exit_time'] = t
                    visitors_in_museum.remove(visitor)

        self.visitor_data = visitor_data
        self.transition_matrix = transition_matrix
        return visitor_data

    def analyze_results(self):
        """Calculate statistics for completed visits."""
        completed_visitors = [v for v in self.visitor_data if v['exit_time'] is not None]
        total_cost = 0

        for visitor in completed_visitors:
            for exhibition_time in visitor['exhibition_times']:
                visitor_cost = ((0.06 * 0.25) + (0.24 * 0.75)) * exhibition_time + \
                               ((exhibition_time * (exhibition_time - 1.0) / 2.0) * 150.0 + 2400.0) * 0.000005
                total_cost += visitor_cost

        avg_cost_per_person = total_cost / len(completed_visitors) if completed_visitors else 0

        stats = {
            'total_visitors': len(self.visitor_data),
            'completed_visits': len(completed_visitors),
            'visit_durations': [v['exit_time'] - v['entry_time'] for v in completed_visitors],
            'exhibition_times': [v['exhibition_times'] for v in completed_visitors],
            'total_cost': total_cost,
            'avg_cost_per_person': avg_cost_per_person
        }

        stats['mean_visit_duration'] = np.mean(stats['visit_durations']) if stats['visit_durations'] else 0
        stats['mean_exhibition_times'] = np.mean(stats['exhibition_times'], axis=0) if stats['exhibition_times'] else None
        return stats

    def plot_transition_matrix(self):
        """Plot the transition matrix as a heatmap."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.transition_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Transition Probability')
        plt.title('Transition Matrix Heatmap')
        plt.xlabel('To Exhibition')
        plt.ylabel('From Exhibition')
        plt.show()

    def plot_results(self, stats):
        """Visualize simulation results."""
        completed_visitors = stats['completed_visits']
        not_completed_visitors = stats['total_visitors'] - completed_visitors

        plt.figure(figsize=(15, 10))

        # Visit duration distribution
        plt.subplot(2, 2, 1)
        visit_durations = np.array(stats['visit_durations'])
        plt.hist(visit_durations, bins=25, color='skyblue', alpha=0.7)
        plt.title('Distribution of Visit Durations')
        plt.xlabel('Visit Duration (minutes)')
        plt.ylabel('Number of Visitors')

        # Exhibition times
        plt.subplot(2, 2, 2)
        exhibition_times = np.array(stats['mean_exhibition_times'])
        if exhibition_times is not None:
            plt.bar(range(len(exhibition_times)), exhibition_times, color='orange', alpha=0.7)
        plt.title('Mean Time Spent at Each Exhibition')
        plt.xlabel('Exhibition Number')
        plt.ylabel('Time (minutes)')

        # Cost distribution
        plt.subplot(2, 2, 3)
        plt.hist([v['exit_time'] - v['entry_time'] for v in self.visitor_data if v['exit_time'] is not None], bins=30)
        plt.title('Cost Distribution')
        plt.xlabel('Cost (per person)')
        plt.ylabel('Frequency')

        # Completion pie chart
        plt.subplot(2, 2, 4)
        plt.pie(
            [completed_visitors, not_completed_visitors],
            labels=['Completed Visits', 'Did Not Complete'],
            autopct='%1.1f%%',
            colors=['gold', 'lightcoral']
        )
        plt.title('Completion Rate')

        plt.tight_layout()
        plt.show()


# Run simulation
sim = MarkovMuseumSimulation(n_visitors=300, n_exhibitions=30, transition_prob=0.05)
visitor_data = sim.simulate()
stats = sim.analyze_results()

# Print statistics
print("\nSimulation Statistics:")
print(f"Total visitors: {stats['total_visitors']}")
print(f"Completed visits: {stats['completed_visits']}")
print(f"Mean visit duration: {stats['mean_visit_duration']:.1f} minutes")
print("Mean time at each exhibition:", stats['mean_exhibition_times'])
print(f"Total cost: {stats['total_cost']:.2f}")
print(f"Average cost per person: {stats['avg_cost_per_person']:.2f}")

# Plot results
sim.plot_results(stats)

# Plot the transition matrix
sim.plot_transition_matrix()
