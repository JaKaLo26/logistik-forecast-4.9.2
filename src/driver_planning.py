"""Explicit driver pool for conservative single-day planning; adapter 1.0."""
import pandas as pd
from src.clustering import DRIVER_DEFAULTS, number

DRIVER_COLUMNS=['driver_id',*DRIVER_DEFAULTS]

def driver_fleet(fleet, driver_rows):
    drivers=pd.DataFrame(driver_rows,columns=DRIVER_COLUMNS)
    if drivers.empty:
        raise ValueError('Mindestens einen verfügbaren Fahrer eintragen.')
    ids=drivers.driver_id.fillna('').astype(str).str.strip()
    if ids.eq('').any() or ids.duplicated().any():
        raise ValueError('Fahrerkennungen müssen gefüllt und eindeutig sein.')
    rows=[]
    for (_, vehicle), (_, driver) in zip(fleet.iterrows(),drivers.iterrows()):
        values={key:number(driver[key],key) for key in DRIVER_DEFAULTS}
        for key,limit in [('remaining_drive_min',540),('remaining_week_drive_min',3360),('remaining_fortnight_drive_min',5400),('drive_since_break_min',270),('work_since_break_min',360),('remaining_duty_min',780)]:
            if values[key]>limit:
                raise ValueError(f'{key}: maximal {limit} Minuten im Standardprofil.')
        if values['remaining_drive_min'] + values['drive_since_break_min'] > 540:
            raise ValueError('Tagesrestlenkzeit und Lenkvorzeit widersprechen dem 9-Stunden-Profil.')
        rows.append({**vehicle.to_dict(),**values,'driver_id':str(driver.driver_id).strip()})
    return pd.DataFrame(rows)
