# 🐜 Max-Min Ant System – begrenztes Pheromon statt unbegrenzter Verstärkung

Zehntes Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Fix-Nachfolger von [ant-system-demo](https://sebastianhanisch-ant-system-demo.streamlit.app/):
dessen "Ehrliche Grenzen" dokumentierten offen die zentrale Schwäche - keine Pheromon-Obergrenze, ein früh gefundener,
mittelmäßiger Pfad kann sich unbegrenzt verstärken, die ganze Kolonie legt sich vorzeitig darauf fest (Stagnation).
Max-Min Ant System (MMAS, Stützle & Hoos, 1996/2000) behebt genau das über zwei Änderungen: Pheromonwerte werden auf
ein Band $[\tau_{min}, \tau_{max}]$ begrenzt, und nur die **beste Ameise** einer Generation legt Pheromon ab (statt
alle). Vehikel ist dieselbe diskrete Lieferroute wie ant-system-demo/genetic-algorithm-demo/nsga2-demo.

## Warum dieses Problem

Ant Systems Stagnationsrisiko ist kein Randfall, sondern eine strukturelle Eigenschaft des unbeschränkten Updates:
je öfter ein Pfad gewählt wird, desto mehr Pheromon bekommt er, desto wahrscheinlicher wird er wieder gewählt - ein
sich selbst verstärkender Kreislauf ohne Bremse. MMAS setzt an genau diesem Mechanismus an, nicht an der
Tourkonstruktion selbst: Übergangswahrscheinlichkeit und Tourbau bleiben identisch zu Ant System, nur das
Pheromon-Update ändert sich.

## Modell

Dieselbe diskrete Lieferroute wie ant-system-demo/genetic-algorithm-demo/nsga2-demo/nsga3-demo/moead-demo: ein Depot
in der Mitte und *n* Kundenstopps in einem 100×100-km-Gebiet. **`mmas_scenario.generate_perm` reproduziert die
Vehikel-Erzeugung wortgleich** (nur die xy-Koordinaten, wie ant-system-demo) - bei Standard-Vehikel-Seed 35 bzw. der
geteilten kleinen Vergleichsinstanz (n=8, Seed 19) bitidentisch zu den Vorgänger-Demos.

## Methodik

Übergangswahrscheinlichkeit und Tourkonstruktion **identisch zu Ant System**: $p_{ij} \propto \tau_{ij}^\alpha
\eta_{ij}^\beta$ ($\eta$ = 1/Distanz). Der Unterschied liegt ausschließlich im Pheromon-Update (`mmas_algorithm.py`,
eigenständig implementiert, kein Cross-Repo-Import):

- **Nur die beste Ameise der Generation legt ab** (iteration-best): $\tau_{ij} \leftarrow (1-\rho)\tau_{ij} +
  \Delta\tau_{ij}^{best}$, $\Delta\tau_{ij}^{best} = Q/L_{best}$ nur für Kanten der besten Tour DIESER Generation.
- **Begrenztes Band**: nach jedem Update $\tau_{ij} \leftarrow \text{clip}(\tau_{ij}, \tau_{min}, \tau_{max})$.
- $\tau_{max} = 1/(\rho \cdot L_{best,global})$ (Literatur-Standardwert, neu berechnet, sobald sich das global beste
  Ergebnis verbessert). $\tau_{min} = \tau_{max} \cdot r$ mit dem τ-Verhältnis $r$ als eigenem Regler.
- Pheromon startet bei $\tau_{max}$ (maximale Anfangsexploration, wie in der Literatur üblich).

**Kreuzprobe**: kein separates externes Referenzpaket (gängige ACO-Pakete wie `acopy` bieten keine ohne Weiteres
konfigurierbare MMAS-Variante) - Rigor über Handrechnungen des Grenzen-Clippings und des Best-Ant-Updates (inkl.
einer Regressionstest gegen einen Diagonalen-Bug, siehe unten) plus denselben Brute-Force-Vergleich auf der kleinen
Vergleichsinstanz wie ant-system-demo (eigenständig neu gemessen, kein Zitat).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Verhindert die Begrenzung wirklich Stagnation? | Ja, klar messbar: bei lockerem τ-Verhältnis (r=0,02, näher am unbeschränkten Ant-System-Verhalten) kleben nach einem langen Lauf 6,67 % der Pheromonwerte nahe τ_max; bei eng gewähltem τ-Verhältnis (r=0,5) sind es 0,0 %. Über 5 verschiedene Vehikel-Seeds exakt reproduzierbar. | `test_stagnation_experiment_headline_claims` |
| Wie stark hängt die Tourqualität vom τ-Verhältnis r ab? | Klarer Trade-off: die mediane Tourlänge steigt von 491 km bei r=0,005 auf 542 km bei r=0,4 - ein zu lockeres Band kostet messbar Qualität. | `test_ratio_experiment_headline_claims` |
| Wie nah kommt MMAS ans echte Optimum? | Auf der kleinen Vergleichsinstanz (8 Stopps) trifft MMAS im Standardfall exakt das Brute-Force-Optimum (255,4 km, 0,0 % Abstand). | `test_kleine_instanz_preset_claims` |
| Wie stark hängt der Abstand zum Optimum von der Ameisenzahl ab? | Wie bei Ant System: bei sehr wenigen Ameisen (4) bleibt ein Abstand zum Optimum, ab etwa 10 Ameisen wird die kleine Vergleichsinstanz praktisch immer exakt gelöst. | `test_ants_sweep_headline_claims` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Vereinfachtes τ_min = τ_max · r** | Stützle & Hoos' Originalarbeit leitet τ_min analytisch über die Konvergenzwahrscheinlichkeit p_best her - diese Demo nutzt bewusst ein festes Verhältnis r statt der vollen Herleitung (dieselbe Art Vereinfachung wie L-SHADEs H-Parameter in dieser Linie). |
| **Keine Stagnations-Erkennung mit Reinitialisierung** | Die vollständige MMAS-Variante erkennt Stagnation aktiv über den λ-Verzweigungsfaktor und setzt Pheromon dann gezielt auf τ_max zurück - hier nicht umgesetzt, die Grenzen allein müssen als Schutz reichen. |
| **Nur iteration-best, keine Ablage-Alternation** | Fortgeschrittenere MMAS-Varianten wechseln zwischen iteration-best- und global-best-Ablage - diese Demo nutzt durchgehend iteration-best. |
| **τ-Verhältnis r ist gut gewählt** | Zu eng: Tourqualität leidet messbar (siehe Befunde oben). Zu locker: Stagnationsrisiko kehrt zurück - muss von Hand eingestellt werden, wie bei jedem Regler dieser Linie. |

Letzter Ast der Ant-System-Kette dieser Linie - kein weiterer Nachfolger geplant.

## Tests

61 Tests (`pytest tests/ -v`): Grenzen-Clipping und Best-Ant-Update per Handrechnung geprüft (inkl. Regressionstest
gegen einen während der Entwicklung gefundenen Bug, bei dem die Diagonale fälschlich mit-geklemmt wurde statt bei 0
zu bleiben), Brute-Force-Vergleich auf sehr kleinen Instanzen, Szenario-Erzeugung bitidentisch zu ant-system-demo
geprüft, AppTest-Rauchtests (jedes Preset, Generation-Slider inkl. Abspielen, Permalink-Grenzen, beide Experimente +
Sweep auf Abruf) und `test_claims.py` (jede Zahl aus diesem README, mit CI-robusten Bändern für
Einzellauf-Kennzahlen - siehe `feedback_ci_platform_robust_tests.md`, von Anfang an angewendet).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `mmas_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `mmas_presets.py` | Permalink/Presets-Mechanik |
| `mmas_scenario.py` | Vehikel-Erzeuger (Lieferroute), wortgleich zu ant-system-demo |
| `mmas_algorithm.py` | MMAS-Kern (Übergangswahrscheinlichkeit, Tourkonstruktion, Best-Ant-Update, Grenzen) |
| `mmas_evaluation.py` | Kennzahlen, Brute-Force-Referenz, Stagnations-Kopfexperiment, τ-Verhältnis-Experiment, Sweep |
| `mmas_visualization.py` | Plotly-Abbildungen (Karte mit pheromonstärke-gewichteten Kanten, τ_min/τ_max-Verlauf, Vergleiche) |

## Bewusst nicht umgesetzt

- Analytische p_best-basierte τ_min-Herleitung (Stützle & Hoos' volle Originalformel).
- λ-Verzweigungsfaktor-Stagnationserkennung mit aktiver Pheromon-Reinitialisierung.
- Alternierende iteration-best/global-best-Ablage.
- Kandidatenlisten oder andere Beschleunigungstechniken für größere Instanzen (wie ant-system-demo).
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
