# REHABILITATION PATIENT SCHEDULING

Die Python Version die wir genutzt haben war 3.10.4. Die benötigten requirements sind in der `requirements.txt`. Den Code kann man starten mit dem Befehl 
```
python -m src "SOLVER" "{}" "INSTANCE_PATH"
```
wo SOLVER den folgenden Wert haben kann: "LBBDSolver", "MIPSolver" und "MIPSolver3", wobei MIPSolver3, das IP-EGA ist.

Um den subsolver auszuwählen, muss der zweite Parameter ersetzt werden mit:
```
{"subsolver_cls":"SUBSOLVER"} 
```
wobei SUBSOLVER den Wert "CPSubsolver" oder "MIPSubsolver" haben kann.

Das zweite Parameter kann auch genutzt werden um weitere Eigenschaften der Solver zu setzen, sollten hoffentlich aber korrekte default werte haben.

Die instanzen die wir gebaut und genutzt haben sind im Order `data/comp_study_004/` zu finden.

Das Cluster Script ist unter `src/cluster/cluster.template` zu finden.
