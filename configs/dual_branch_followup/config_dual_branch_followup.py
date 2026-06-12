RUNS_DIR_NAME = "run14"

EXPERIMENT_SPECS = {
    "E1": {
        "fusion_mode": "shallow_gate",
        "boundary_loss_weight": 0.0,
        "description": "Gate scales 1-3; disable edge injection at scales 4-5.",
    },
    "E2": {
        "fusion_mode": "multiscale_gate",
        "boundary_loss_weight": 0.1,
        "description": "D3 multiscale gate with boundary loss weight 0.1.",
    },
    "E3": {
        "fusion_mode": "multiscale_gate",
        "boundary_loss_weight": 0.2,
        "description": "D3 multiscale gate with boundary loss weight 0.2.",
    },
}
