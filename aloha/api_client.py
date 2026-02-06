# %%
import requests

BASE_URL = "http://localhost:8801"

# %%
response = requests.get(f"{BASE_URL}/env")
env_data = response.json()
print(env_data)

# %%
obj_pos = env_data['objects']['object_red_0']['pos']

pick_code = f"pick_object({obj_pos}, verbose=True)"
payload = {
    "action": {
        "type": "run_code",
        "payload": {"code": pick_code}
    }
}
response = requests.post(f"{BASE_URL}/send_action", json=payload)
print(response.json())

# %%
# Move left arm EE to table center (prepare to receive cube from right arm)
# Table center is approximately at (0, 0, 0.02) - slightly above table surface
table_center_pos = [0.0, 0.0, 0.3]  # 5cm above table to avoid collision

move_code = f"set_ee_target_position({table_center_pos}, timeout=10.0, verbose=True)"
payload = {
    "action": {
        "type": "run_code",
        "payload": {"code": move_code}
    }
}
response = requests.post(f"{BASE_URL}/send_action", json=payload)
print(response.json())

# %%
