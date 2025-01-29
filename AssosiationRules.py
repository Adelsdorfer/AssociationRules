import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import hashlib
import re

# Pfad zur Excel-Datei
file_path = r'C:\\Assoziationsanalyse\\SPC.xlsx' 

# Laden der Excel-Datei (erstes Blatt)
df = pd.read_excel(file_path, engine='openpyxl')

# Überprüfen, ob mindestens drei Spalten vorhanden sind (z.B. ID, Artikel, Bestellnummer)
if df.shape[1] < 3:
    raise ValueError("Die Excel-Datei muss mindestens drei Spalten enthalten: "
                     "z.B. Transaktions-ID, Artikel und Bestellnummer.")

# -- 1) Identifiziere die relevanten Spalten ----------------------------------

# Beispielhafte Annahme:
#   - Erste Spalte: Transaktions-ID (id_column)
#   - Zweite Spalte: Artikelname (item_column)
#   - Dritte Spalte: Bestellnummer (bestellnummer_column)
id_column = df.columns[0]
item_column = df.columns[1]
bestellnummer_column = df.columns[2]

print(f"Identifizierte Transaktions-ID-Spalte: {id_column}")
print(f"Identifizierte Artikelspalte: {item_column}")
print(f"Identifizierte Bestellnummer-Spalte: {bestellnummer_column}")

# -- 2) Erstelle ein Mapping Artikel -> Bestellnummer -------------------------

unique_items = df[[item_column, bestellnummer_column]].drop_duplicates()
item_bestell_map = dict(zip(unique_items[item_column], unique_items[bestellnummer_column]))

# -- 3) Transaktionsdaten aufbereiten (Artikel pro Warenkorb) -----------------

grouped = df.groupby(id_column)[item_column].apply(
    lambda x: [str(item).strip() for item in x if pd.notna(item)]
).reset_index(name='Items')

transactions = grouped['Items'].tolist()
if not transactions:
    raise ValueError("Keine Transaktionen gefunden. Bitte überprüfen Sie die Daten.")

# -- 4) TransactionEncoder anwenden -------------------------------------------

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

# -- 5) Apriori und Assoziationsregeln ----------------------------------------

frequent_itemsets = apriori(df_transformed, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.15)

# -- 6) combination_count -----------------------------------------------------

rules['combination_count'] = rules.apply(
    lambda row: df_transformed.loc[
        (df_transformed[list(row['antecedents'])].all(axis=1)) &
        (df_transformed[list(row['consequents'])].all(axis=1))
    ].shape[0],
    axis=1
)

# -- 7) Bestellnummern für Antecedents und Consequents ermitteln (Mat_combination)

def get_mat_combination(row):
    """
    Liefert nur die Bestellnummern der Artikel in Antecedent + Consequent,
    getrennt durch '-'.
    """
    items_in_rule = row['antecedents'].union(row['consequents'])
    bestellnummern = []
    
    for item in items_in_rule:
        if item in item_bestell_map:
            bestellnummern.append(str(item_bestell_map[item]))
    
    # Sortierte, eindeutige Bestellnummern per '-' verbinden
    bestellnummern = sorted(set(bestellnummern))
    return '-'.join(bestellnummern)

rules['Mat_combination'] = rules.apply(get_mat_combination, axis=1)

# -- 8) Eindeutige 8-stellige ID erzeugen (ohne führende 0 -> 9) --------------

def create_unique_8digit_id(combination_string):
    """
    Erzeugt aus dem SHA-256-Hash des combination_string eine 8-stellige Zahl,
    wobei führende '0' durch '9' ersetzt werden.
    """
    # 1) SHA-256-Hash berechnen
    sha_value = hashlib.sha256(combination_string.encode("utf-8")).hexdigest()
    
    # 2) Hash in Integer umwandeln
    hash_int = int(sha_value, 16)
    
    # 3) Modulo 10^8 -> Wert zwischen 0 und 99.999.999
    mod_value = hash_int % 10**8
    
    # 4) Auf 8 Zeichen auffüllen (zfill) und ...
    mod_str = str(mod_value).zfill(8)
    
    # 5) ... alle führenden '0' durch '9' ersetzen
    #    (z.B. '00012345' -> '99912345'; '00000000' -> '99999999')
    #    per Regex: alle aufeinanderfolgenden '0' am Anfang durch
    #    gleich viele '9' ersetzen
    mod_str = re.sub(r'^(0+)', lambda m: '9' * len(m.group(1)), mod_str)
    
    return mod_str

rules['Mat_combination_id'] = rules['Mat_combination'].apply(create_unique_8digit_id)

# -- 9) Spalte 'different items' ----------------------------------------------

rules['different items'] = rules.apply(
    lambda row: len(row['antecedents'].union(row['consequents'])), 
    axis=1
)

# -- 10) Antezedenten und Konsequenten in Strings umwandeln -------------------

rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

# -- 11) Export der Ergebnisse nach Excel -------------------------------------

output_file_path = r'C:\\Assoziationsanalyse\\SPC_Regeln.xlsx'
rules_export = rules[[
    'antecedents', 
    'consequents', 
    'support', 
    'confidence', 
    'lift', 
    'leverage', 
    'conviction', 
    'zhangs_metric', 
    'combination_count', 
    'Mat_combination',    
    'Mat_combination_id',  # 8-stellig, führende '0' → '9'
    'different items'
]]
rules_export.to_excel(output_file_path, index=False)

print(f"Assoziationsregeln wurden erfolgreich in '{output_file_path}' exportiert.")
print(rules_export)

# -- 12) Visualisierung als Link-Graph ----------------------------------------

G = nx.DiGraph()

# Knoten und Kanten basierend auf Regeln hinzufügen
for _, row in rules.iterrows():
    G.add_edge(
        row['antecedents'], 
        row['consequents'], 
        weight=row['confidence'], 
        combination_count=row['combination_count']
    )

# Knotengrößen basierend auf combination_count berechnen
node_sizes = {}
for node in G.nodes():
    total_count = rules.apply(
        lambda row: row['combination_count'] 
        if (node in row['antecedents']) or (node in row['consequents']) 
        else 0,
        axis=1
    ).sum()
    node_sizes[node] = total_count

# Normalisierung für die Darstellung
min_size = 300
max_size = 3000
counts = list(node_sizes.values())
if counts:
    min_count = min(counts)
    max_count = max(counts)
    if max_count == min_count:
        scaled_sizes = {node: (min_size + max_size) / 2 for node in node_sizes}
    else:
        scaled_sizes = {
            node: min_size + (size - min_count) / (max_count - min_count) * (max_size - min_size)
            for node, size in node_sizes.items()
        }
else:
    scaled_sizes = {node: min_size for node in node_sizes}

# Zeichnen des Graphen
pos = nx.spring_layout(G, dim=2, k=0.3, scale=20.0, center=None, iterations=100)
edges = G.edges(data=True)

# Kanten
nx.draw_networkx_edges(
    G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=7,
    edge_color=[d['weight'] for (u, v, d) in edges],
    edge_cmap=plt.cm.Blues, width=2
)

# Knoten
nx.draw_networkx_nodes(
    G, pos,
    node_size=[scaled_sizes[node] for node in G.nodes()],
    node_color='skyblue'
)

# Labels
nx.draw_networkx_labels(
    G, pos, font_size=9, font_color='purple'
)

# Edge-Labels (Konfidenzwerte)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels,
    font_color='red', font_size=8
)

plt.title('Assoziationsregeln Link-Graph (basierend auf Konfidenzniveau)')
plt.axis('off')  # Achsen ausgeblendet für eine sauberere Darstellung
plt.show()
