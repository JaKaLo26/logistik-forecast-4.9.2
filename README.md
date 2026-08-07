title: LagerForecast 4.9.3 emoji: 🚚 colorFrom: blue colorTo: green sdk: gradio sdk_version: 5.49.1 python_version: "3.11" app_file: app.py pinned: false
LagerForecast 4.9.3
Version 4.9.3 setzt die neue verbindliche Prozessstruktur um:
CSV importieren.
Adressen automatisch prüfen.
Nur unsichere Adressen nacheinander bestätigen.
Depot und verfügbare Flotte festlegen.
Aufträge geografisch clustern.
Cluster unter Paletten- und Nutzlastgrenzen LKW zuweisen.
Erst danach die Stopp-Reihenfolge je LKW über OSRM Trip optimieren.
Erst die fertige Route auf Verkehr/Störungen prüfen.
Forecast und Kartenansicht je LKW anzeigen.
Wesentliche Änderung zu 4.9.2
Die Auftragsverteilung arbeitet nicht mehr nach dem Prinzip „was noch hineinpasst“. Stattdessen werden räumlich nahe Stopps zu zusammenhängenden Liefergebieten gebündelt. Dadurch sollen sich Touren weniger überschneiden und jeder LKW möglichst wenig unnötige Strecke zwischen Regionen fahren.
Clusterheuristik
Farthest-first Seed ab Depot
Near-neighbour Region Growing
harte Grenzen für Palettenplätze und Nutzlast
leichte Berücksichtigung der Zeitfenster-Mittelpunkte
kompakte, möglichst nicht überlappende Liefergebiete
Routenoptimierung
Nach der Clusterbildung wird pro Fahrzeug der OSRM-Trip-Service verwendet, um die Reihenfolge der Stopps als Rundtour ab/bis Depot zu optimieren.
Verkehr / Forecast
Verkehr wird bewusst erst nach der Routenoptimierung ausgewertet. Die offizielle Autobahn-API ist als Adapter vorhanden. Die produktive räumliche Zuordnung von Autobahn-Ereignissen zur optimierten Route ist der nächste Entwicklungsschritt und wird nicht durch künstliche Baustellenindizes ersetzt.
Start
pip install -r requirements.txt
python app.py