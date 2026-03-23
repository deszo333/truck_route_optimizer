import pandas as pd

def inject_urgent_request(existing_plan, new_request, trucks_df, dist_dict):
    """Smart dynamic insertion based on capacity and distance."""
    region_plan = existing_plan[existing_plan['region'] == new_request['region']]
    if region_plan.empty: return existing_plan
    
    # Calculate remaining capacity
    truck_loads = region_plan.groupby('truck_id')['weight_kg'].sum().to_dict()
    truck_capacities = trucks_df.set_index('truck_id')['capacity_kg'].to_dict()
    
    best_truck = None
    best_cost = float('inf')
    last_sequence = 0
    
    for truck_id, current_load in truck_loads.items():
        max_cap = truck_capacities.get(truck_id, 0)
        if (max_cap - current_load) >= new_request['total_weight_kg']:
            
            truck_route = region_plan[region_plan['truck_id'] == truck_id]
            last_stop = truck_route.iloc[-1]['destination']
            
            detour_cost = dist_dict.get((last_stop, new_request['origin_branch']), 20.0)
            
            if detour_cost < best_cost:
                best_cost = detour_cost
                best_truck = truck_id
                last_sequence = truck_route.iloc[-1]['stop_sequence']

    if best_truck:
        new_leg = pd.DataFrame([{
            'region': new_request['region'],
            'truck_id': best_truck,
            'stop_sequence': last_sequence + 1,
            'request_id': new_request['transaction_id'] + " (URGENT)",
            'origin': new_request['origin_branch'],
            'destination': new_request['destination_branch'],
            'weight_kg': new_request['total_weight_kg'],
            'deadline_hrs': 1,
            'estimated_distance_km': dist_dict.get((new_request['origin_branch'], new_request['destination_branch']), 50.0)
        }])
        return pd.concat([existing_plan, new_leg], ignore_index=True)
        
    return existing_plan