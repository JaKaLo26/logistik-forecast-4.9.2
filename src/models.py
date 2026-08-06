from dataclasses import dataclass, asdict

@dataclass
class Vehicle:
    vehicle_id: str
    vehicle_class: str
    pallet_capacity: int
    payload_kg: int
    color: str
    available: bool = True

    def to_dict(self):
        return asdict(self)
