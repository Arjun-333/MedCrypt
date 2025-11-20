from fastapi import FastAPI
from pydantic import BaseModel
from fl_simulation import (
    SimpleCNN,
    synthetic_client_data,
    local_train,
    aggregate_weights,
    evaluate
)
import threading

app = FastAPI(title="MedCrypt Federated Server")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins (for local development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL MODEL (one instance)
global_model = SimpleCNN()

# Simulate 3 hospitals with separate synthetic datasets
clients = [synthetic_client_data(seed=i) for i in range(3)]

# Thread lock for safe updates
lock = threading.Lock()

class TrainRequest(BaseModel):
    rounds: int = 1
    epochs: int = 1
    dp_noise_std: float = 0.01

@app.get("/ping")
def ping():
    return {"message": "MedCrypt FL Server is running"}

@app.post("/train_round")
def train_round(req: TrainRequest):
    with lock:
        results = []
        for r in range(req.rounds):
            local_states = []
            for i, data in enumerate(clients):
                sd = local_train(global_model, data, epochs=req.epochs, dp_noise_std=req.dp_noise_std)
                local_states.append(sd)

            # FedAvg aggregation
            merged = aggregate_weights(local_states)
            global_model.load_state_dict(merged)

            # Evaluations
            accs = [evaluate(global_model.state_dict(), SimpleCNN, d) for d in clients]
            results.append({"round": r+1, "accs": accs})

        return {"status": "success", "details": results}

@app.get("/accuracy")
def accuracy():
    test_x, test_y = synthetic_client_data(num_samples=300, seed=99)
    acc = evaluate(global_model.state_dict(), SimpleCNN, (test_x, test_y))
    return {"global_accuracy": acc}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#             sd = local_train(global_model, client_data[i], epochs=1, lr=0.01, dp_noise_std=0.01, device=device)
#             sds.append(sd)