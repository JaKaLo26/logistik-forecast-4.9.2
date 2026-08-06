from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json, math, requests

@dataclass
class TrafficResult:
    provider: str
    delay_s: float
    score: float
    confidence: float
    incidents: list[dict]
    debug: dict

class HereTrafficProvider:
    def __init__(self, api_key: str, timeout: int = 20):
        self.api_key = api_key.strip()
        self.timeout = timeout

    def available(self): return bool(self.api_key)

    def analyze_route(self, geometry: list[tuple[float,float]], baseline_duration_s: float) -> TrafficResult:
        if not self.available() or not geometry:
            return TrafficResult('HERE',0,0,0,[],{'status':'skipped','reason':'kein API-Key oder keine Geometrie'})
        sample = geometry[::max(1, len(geometry)//12)][:12]
        jams, speeds, raws, errors = [], [], [], []
        for lat, lon in sample:
            params = {'in':f'circle:{lat},{lon};r=800','locationReferencing':'shape','apiKey':self.api_key}
            try:
                r = requests.get('https://data.traffic.hereapi.com/v7/flow', params=params, timeout=self.timeout)
                raw = {'url': r.url.replace(self.api_key,'***'), 'status_code': r.status_code}
                r.raise_for_status(); data = r.json(); raw['response'] = data; raws.append(raw)
                for item in data.get('results',[]):
                    current = item.get('currentFlow',{})
                    if current.get('jamFactor') is not None: jams.append(float(current['jamFactor']))
                    if current.get('speed') is not None: speeds.append(float(current['speed']))
            except Exception as e: errors.append(str(e))
        jam = sum(jams)/len(jams) if jams else 0
        score = min(100, jam*10)
        delay = baseline_duration_s * min(1.5, jam/18) if jam > 0 else 0
        conf = min(1, len(jams)/max(1,len(sample)))
        return TrafficResult('HERE',delay,score,conf,[],{
            'sample_points':len(sample),'jam_values':jams,'speed_values':speeds,'requests':raws,'errors':errors,
            'mapping':{'jamFactor':'0=frei, 10=starker Stau/nahe Sperrung','speed':'aktuelle Geschwindigkeit'},
        })

class AutobahnProvider:
    def __init__(self, base_url='https://verkehr.autobahn.de/o/autobahn', timeout=20):
        self.base_url=base_url.rstrip('/'); self.timeout=timeout
    def analyze_route(self, geometry, baseline_duration_s):
        # Kein verlässlicher allgemeiner Straßenname aus Geometrie allein: bewusst nur Debug/Adapterstatus.
        return TrafficResult('Autobahn API',0,0,0,[],{
            'status':'prepared','note':'Für produktive Abfragen werden aus OSRM-Schritten erkannte A-Nummern benötigt.'
        })

def combine(results: list[TrafficResult], weights: dict[str,float]) -> dict:
    active=[r for r in results if r.confidence>0]
    if not active:
        return {'delay_s':0,'score':0,'confidence':0,'provider_results':[r.__dict__ for r in results]}
    denom=sum(weights.get(r.provider,1) for r in active)
    score=sum(r.score*weights.get(r.provider,1) for r in active)/denom
    delay=sum(r.delay_s*weights.get(r.provider,1) for r in active)/denom
    confidence=sum(r.confidence*weights.get(r.provider,1) for r in active)/denom
    return {'delay_s':delay,'score':score,'confidence':confidence,'provider_results':[r.__dict__ for r in results]}
