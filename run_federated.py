from fl_simulation import (
    SimpleCNN,
    synthetic_client_data,
    local_train,
    aggregate_weights,
    evaluate
)

def simulate_federated_rounds(num_clients=3, rounds=3, dp_noise=0.01):
    # Initialize global model
    global_model = SimpleCNN()

    # Create separate synthetic datasets for each "hospital"
    clients = [synthetic_client_data(seed=i) for i in range(num_clients)]

    print("\n=== Federated Learning Simulation ===\n")

    for r in range(rounds):
        print(f"--- Round {r+1} ---")
        local_states = []

        # Local training per hospital
        for i, data in enumerate(clients):
            print(f"Training Hospital {i+1}...")
            sd = local_train(global_model, data, epochs=1, dp_noise_std=dp_noise)
            local_states.append(sd)

        # FedAvg aggregation
        agg = aggregate_weights(local_states)
        global_model.load_state_dict(agg)

        # Show accuracy per hospital
        accs = [evaluate(global_model.state_dict(), SimpleCNN, d) for d in clients]
        print("Accuracies:", accs)
        print()

if __name__ == "__main__":
    simulate_federated_rounds()
