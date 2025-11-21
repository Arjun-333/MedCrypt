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

# State to track simulation settings
simulation_state = {
    "non_iid": False,
    "clients": [synthetic_client_data(seed=i, non_iid=False, client_id=i) for i in range(3)]
}

# Thread lock for safe updates
lock = threading.Lock()

class TrainRequest(BaseModel):
    rounds: int = 1
    epochs: int = 1
    dp_noise_std: float = 0.01
    mu: float = 0.0   # FedProx parameter
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

# State to track simulation settings
simulation_state = {
    "non_iid": False,
    "clients": [synthetic_client_data(seed=i, non_iid=False, client_id=i) for i in range(3)]
}

# Thread lock for safe updates
lock = threading.Lock()

class TrainRequest(BaseModel):
    rounds: int = 1
    epochs: int = 1
    dp_noise_std: float = 0.01
    mu: float = 0.0   # FedProx parameter
    non_iid: bool = False

@app.get("/ping")
def ping():
    return {"message": "MedCrypt FL Server is running"}

@app.post("/reset")
def reset_simulation():
    global global_model
    with lock:
        global_model = SimpleCNN()
        # Reset clients to default IID
        simulation_state["non_iid"] = False
        simulation_state["clients"] = [synthetic_client_data(seed=i, non_iid=False, client_id=i) for i in range(3)]
    return {"message": "Simulation reset successfully"}

import uuid
from fastapi import BackgroundTasks

# Store tasks: task_id -> {"status": "running"|"completed"|"failed", "result": ...}
tasks = {}

def run_training_background(task_id: str, req: TrainRequest):
    try:
        with lock:
            # Check if we need to regenerate data for Non-IID switch
            if req.non_iid != simulation_state["non_iid"]:
                print(f"Switching data distribution to Non-IID={req.non_iid}")
                simulation_state["non_iid"] = req.non_iid
                simulation_state["clients"] = [
                    synthetic_client_data(seed=i, non_iid=req.non_iid, client_id=i) 
                    for i in range(3)
                ]

            results = []
            for r in range(req.rounds):
                local_states = []
                for i, data in enumerate(simulation_state["clients"]):
                    sd = local_train(
                        global_model, 
                        data, 
                        epochs=req.epochs, 
                        dp_noise_std=req.dp_noise_std,
                        mu=req.mu
                    )
                    local_states.append(sd)

                # FedAvg aggregation
                merged = aggregate_weights(local_states)
                global_model.load_state_dict(merged)

                # Evaluations
                accs = [evaluate(global_model.state_dict(), SimpleCNN, d) for d in simulation_state["clients"]]
                results.append({"round": r+1, "accs": accs})
            
            tasks[task_id] = {"status": "completed", "details": results}
    except Exception as e:
        print(f"Training failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/train_round")
def train_round(req: TrainRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "running"}
    background_tasks.add_task(run_training_background, task_id, req)
    return {"task_id": task_id, "status": "started"}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})

@app.get("/accuracy")
def accuracy():
    # Test on a held-out IID dataset to see global generalization
    test_x, test_y = synthetic_client_data(num_samples=300, seed=99, non_iid=False)
    acc = evaluate(global_model.state_dict(), SimpleCNN, (test_x, test_y))
    return {"global_accuracy": acc}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)