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
        d = asdict(self)
        # UI verwendet "class", damit die Tabelle kompakter bleibt.
        d["class"] = d.pop("vehicle_class")
        return d
