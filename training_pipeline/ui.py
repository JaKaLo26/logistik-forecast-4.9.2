"""Start separately: python -m training_pipeline.ui"""
from pathlib import Path
from uuid import uuid4
import json
import shutil
import gradio as gr
from .core import Config, generate, train


def run(tours,low,high,seed,mode,addresses,source):
    root=Path('pipeline_runs')/uuid4().hex
    root.mkdir(parents=True)
    if mode=='Adresspool + OSRM' and (not addresses or not source):
        raise gr.Error('Adresspool und Quellennachweis hochladen.')
    try:
        data=generate(root/'data',Config(tours=int(tours),min_stops=int(low),max_stops=int(high),seed=int(seed)),
                      addresses if mode=='Adresspool + OSRM' else None,
                      source if mode=='Adresspool + OSRM' else None)
        report=train(data/'segments.csv',root/'model',int(seed))
        archive=shutil.make_archive(str(root/'experiment'),'zip',root_dir=root/'data')
        bundle=shutil.make_archive(str(root/'trained_model'),'zip',root_dir=root/'model')
        return 'Fertig: Modell trainiert, gespeichert, erneut geladen und auf Testtouren geprüft.',json.dumps(report,indent=2,ensure_ascii=False),[archive,bundle]
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

with gr.Blocks(title='Logistik · Trainingslabor',theme=gr.themes.Soft(primary_hue='sky')) as demo:
    gr.Markdown('# Logistik · Trainingslabor\nEigenständiger Machbarkeitsnachweis: Touren → Training → Modell. **Alle Zielzeiten sind synthetisch.**')
    mode=gr.Radio(['Offline-Demo','Adresspool + OSRM'],value='Offline-Demo',label='Datengrundlage')
    gr.Markdown('Offline-Demo: erfundene Adressen und Strecken, nicht zur Adressprüfung verwenden. Adresspool: geprüfte Deutschland-Adressen; OSRM liefert Straßenrouten ohne LKW-Beschränkungsprüfung. Der erste Poolpunkt ist nicht automatisch ein festes Depot: jeder Tourstart wird zufällig gewählt.')
    with gr.Row():
        tours=gr.Number(value=100,label='Anzahl Touren',precision=0,minimum=10)
        low=gr.Number(value=3,label='Stopps von',precision=0,minimum=1)
        high=gr.Number(value=8,label='Stopps bis',precision=0,minimum=1)
        seed=gr.Number(value=42,label='Zufallsstartwert',precision=0,minimum=0)
    addresses=gr.File(label='Adresspool CSV (optional im Demomodus)',type='filepath',file_types=['.csv'])
    source=gr.File(label='Quellennachweis JSON',type='filepath',file_types=['.json'])
    button=gr.Button('Touren erzeugen & Modell trainieren',variant='primary')
    status=gr.Markdown()
    report=gr.Code(label='Split, Kennzahlen und Nachweise',language='json')
    files=gr.File(label='Daten und fertiges Modell herunterladen',file_count='multiple')
    button.click(run,[tours,low,high,seed,mode,addresses,source],[status,report,files])
if __name__=='__main__':
    demo.queue().launch(server_name='0.0.0.0',server_port=7861)
