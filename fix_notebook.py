import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/etapa2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ─── Cell 41 (index 40): Correcciones D2 con limpieza de invalidos ───
cells[40]['source'] = [
    "n0 = len(df2)\n"
    "df2 = df2.drop_duplicates().reset_index(drop=True)\n"
    "print(f'Duplicados eliminados D2: {n0 - len(df2)}')\n"
    "\n"
    "# Conversion de tipos numericos antes de comparar rangos\n"
    "for col in ['edad_madre', 'peso_rn', 'imc_pregestacional']:\n"
    "    if col in df2.columns:\n"
    "        df2[col] = pd.to_numeric(df2[col], errors='coerce')\n"
    "\n"
    "if 'edad_madre' in df2.columns:\n"
    "    mask = df2['edad_madre'].notna() & ((df2['edad_madre'] < 15) | (df2['edad_madre'] > 55))\n"
    "    print(f'Edad madre fuera de [15,55]: {mask.sum()}')\n"
    "    if mask.sum() > 0:\n"
    "        display(df2.loc[mask, ['edad_madre']])\n"
    "        df2.loc[mask, 'edad_madre'] = np.nan\n"
    "\n"
    "if 'peso_rn' in df2.columns:\n"
    "    mask = df2['peso_rn'].notna() & ((df2['peso_rn'] < 400) | (df2['peso_rn'] > 6000))\n"
    "    print(f'Peso RN fuera de [400,6000] g: {mask.sum()}')\n"
    "    if mask.sum() > 0:\n"
    "        display(df2.loc[mask, ['peso_rn']])\n"
    "        df2.loc[mask, 'peso_rn'] = np.nan\n"
    "\n"
    "if 'imc_pregestacional' in df2.columns:\n"
    "    mask = df2['imc_pregestacional'].notna() & ((df2['imc_pregestacional'] < 14) | (df2['imc_pregestacional'] > 60))\n"
    "    print(f'IMC fuera de [14,60]: {mask.sum()}')\n"
    "    if mask.sum() > 0:\n"
    "        display(df2.loc[mask, ['imc_pregestacional']])\n"
    "        df2.loc[mask, 'imc_pregestacional'] = np.nan\n"
    "\n"
    "# Eliminar filas con valores invalidos en variables categoricas\n"
    "# (filas de estadisticas resumen embebidas en el Excel: 847.0, decimales, etc.)\n"
    "valid_sexo  = {1.0, 2.0}\n"
    "valid_bin   = {0.0, 1.0}\n"
    "valid_neduc = {1.0, 2.0, 3.0, 4.0, 5.0}\n"
    "\n"
    "mask_inv = pd.Series(False, index=df2.index)\n"
    "for col, validos in [\n"
    "    ('sexo_rn_c',           valid_sexo),\n"
    "    ('primipara_c',         valid_bin),\n"
    "    ('pandemia_c',          valid_bin),\n"
    "    ('nivel_educacional_c', valid_neduc),\n"
    "]:\n"
    "    if col in df2.columns:\n"
    "        s = pd.to_numeric(df2[col], errors='coerce')\n"
    "        mask_inv |= s.notna() & ~s.isin(validos)\n"
    "\n"
    "print(f'Filas con valores invalidos en categoricas: {mask_inv.sum()} -> eliminadas')\n"
    "if mask_inv.sum() > 0:\n"
    "    cols_show = [c for c in ['sexo_rn_c','nivel_educacional_c','primipara_c','pandemia_c'] if c in df2.columns]\n"
    "    display(df2.loc[mask_inv, cols_show])\n"
    "df2 = df2[~mask_inv].reset_index(drop=True)\n"
    "\n"
    "print(f'\\nTotal registros Dataset 2 tras limpieza: {len(df2)}')\n"
    "print('\\nCorrecciones Dataset 2 completadas.')\n"
]
cells[40]['outputs'] = []
cells[40]['execution_count'] = None
print("c40 OK")

# ─── Cell 52 (index 51): Boxplots D1 sin uf_af ni paridad ───
cells[51]['source'] = [
    "vars_box_d1 = ['edad', 'pnacer', 'edad_gestacion_ultimo_hijo', 'n__panes']\n"
    "vars_box_d1 = [v for v in vars_box_d1 if v in df1.columns and df1[v].notna().sum() > 5]\n"
    "\n"
    "etiq_d1 = {\n"
    "    'edad': 'Edad materna (anios)', 'pnacer': 'Peso al nacer (g)',\n"
    "    'edad_gestacion_ultimo_hijo': 'Edad gestacional (sem)',\n"
    "    'n__panes': 'N panes/dia',\n"
    "}\n"
    "\n"
    "n = len(vars_box_d1)\n"
    "fig, axes = plt.subplots(1, n, figsize=(4*n, 5))\n"
    "if n == 1: axes = [axes]\n"
    "\n"
    "for ax, col in zip(axes, vars_box_d1):\n"
    "    data = pd.to_numeric(df1[col], errors='coerce').dropna()\n"
    "    ax.boxplot(data, patch_artist=True, widths=0.5,\n"
    "               boxprops=dict(facecolor='#4C72B0', alpha=0.7),\n"
    "               medianprops=dict(color='red', linewidth=2),\n"
    "               whiskerprops=dict(linestyle='--'),\n"
    "               flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))\n"
    "    ax.set_title(etiq_d1.get(col, col), fontsize=9)\n"
    "    ax.set_ylabel('Valor')\n"
    "    ax.set_xticks([])\n"
    "\n"
    "fig.suptitle('Boxplots - Dataset 1: Ingesta de AF', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(OUT_DIR, 'boxplots_dataset1.png'), bbox_inches='tight', dpi=100)\n"
    "plt.show()\n"
]
cells[51]['outputs'] = []
cells[51]['execution_count'] = None
print("c51 OK")

# ─── Cell 54 (index 53): Histogramas D1 continuas + pie discretas ───
cells[53]['source'] = [
    "# Histogramas para variables continuas\n"
    "vars_hist_d1 = ['edad', 'pnacer', 'edad_gestacion_ultimo_hijo', 'n__panes']\n"
    "vars_hist_d1 = [v for v in vars_hist_d1 if v in df1.columns and df1[v].notna().sum() > 5]\n"
    "\n"
    "titulos_hist_d1 = {\n"
    "    'edad': 'Edad materna (anios)', 'pnacer': 'Peso al nacer (g)',\n"
    "    'edad_gestacion_ultimo_hijo': 'Edad gestacional (sem)',\n"
    "    'n__panes': 'N panes/dia',\n"
    "}\n"
    "\n"
    "nc = 2\n"
    "nr = (len(vars_hist_d1) + nc - 1) // nc\n"
    "fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 4*nr))\n"
    "axes = axes.flatten()\n"
    "\n"
    "for i, col in enumerate(vars_hist_d1):\n"
    "    data = pd.to_numeric(df1[col], errors='coerce').dropna()\n"
    "    axes[i].hist(data, bins=22, color='#4C72B0', edgecolor='white', alpha=0.85)\n"
    "    axes[i].axvline(data.mean(),   color='red',   linestyle='--', label=f'Media: {data.mean():.1f}')\n"
    "    axes[i].axvline(data.median(), color='green', linestyle='--', label=f'Mediana: {data.median():.1f}')\n"
    "    axes[i].set_title(titulos_hist_d1.get(col, col), fontsize=9)\n"
    "    axes[i].set_xlabel('Valor')\n"
    "    axes[i].set_ylabel('Frecuencia')\n"
    "    axes[i].legend(fontsize=7)\n"
    "\n"
    "for j in range(len(vars_hist_d1), len(axes)):\n"
    "    axes[j].set_visible(False)\n"
    "\n"
    "fig.suptitle('Histogramas - Dataset 1: Variables continuas', fontsize=14)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(OUT_DIR, 'histogramas_dataset1.png'), bbox_inches='tight', dpi=100)\n"
    "plt.show()\n"
    "\n"
    "# Pie charts para variables discretas: Dosis AF y Paridad\n"
    "discretas_d1 = []\n"
    "for col, titulo in [('uf_af', 'Dosis suplemento AF (mcg)'), ('paridad', 'Paridad (N hijos previos)')]:\n"
    "    if col in df1.columns and df1[col].notna().sum() > 5:\n"
    "        discretas_d1.append((col, titulo))\n"
    "\n"
    "if discretas_d1:\n"
    "    fig2, axes2 = plt.subplots(1, len(discretas_d1), figsize=(5*len(discretas_d1), 5))\n"
    "    if len(discretas_d1) == 1: axes2 = [axes2]\n"
    "    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']\n"
    "    for ax, (col, titulo) in zip(axes2, discretas_d1):\n"
    "        s = pd.to_numeric(df1[col], errors='coerce').dropna()\n"
    "        freq = s.value_counts().sort_index()\n"
    "        labels = [str(int(v)) if v == int(v) else str(round(v,1)) for v in freq.index]\n"
    "        ax.pie(freq.values, labels=labels, autopct='%1.1f%%',\n"
    "               colors=colors[:len(freq)], startangle=90,\n"
    "               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})\n"
    "        ax.set_title(titulo, fontsize=11)\n"
    "    fig2.suptitle('Distribucion de variables discretas - Dataset 1', fontsize=13)\n"
    "    plt.tight_layout()\n"
    "    plt.savefig(os.path.join(OUT_DIR, 'pie_discretas_dataset1.png'), bbox_inches='tight', dpi=100)\n"
    "    plt.show()\n"
]
cells[53]['outputs'] = []
cells[53]['execution_count'] = None
print("c53 OK")

# ─── Cell 56 (index 55): Boxplots D2 solo 3 variables ───
cells[55]['source'] = [
    "vars_num_d2 = ['edad_madre', 'peso_rn', 'imc_pregestacional']\n"
    "vars_num_d2 = [v for v in vars_num_d2 if v in df2.columns and df2[v].notna().sum() > 5]\n"
    "\n"
    "titulos_d2 = {\n"
    "    'edad_madre': 'Edad materna (anios)', 'peso_rn': 'Peso RN (g)',\n"
    "    'imc_pregestacional': 'IMC pregestacional',\n"
    "}\n"
    "\n"
    "n = len(vars_num_d2)\n"
    "fig, axes = plt.subplots(1, n, figsize=(4*n, 5))\n"
    "if n == 1: axes = [axes]\n"
    "\n"
    "for ax, col in zip(axes, vars_num_d2):\n"
    "    data = pd.to_numeric(df2[col], errors='coerce').dropna()\n"
    "    ax.boxplot(data, patch_artist=True,\n"
    "               boxprops=dict(facecolor='#DD8452', alpha=0.7),\n"
    "               medianprops=dict(color='red', linewidth=2),\n"
    "               flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))\n"
    "    ax.set_title(titulos_d2.get(col, col), fontsize=9)\n"
    "    ax.set_xticks([])\n"
    "    ax.set_ylabel('Valor')\n"
    "\n"
    "fig.suptitle('Boxplots - Dataset 2: Encuesta AF Completa', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(OUT_DIR, 'boxplots_dataset2.png'), bbox_inches='tight', dpi=100)\n"
    "plt.show()\n"
]
cells[55]['outputs'] = []
cells[55]['execution_count'] = None
print("c55 OK")

# ─── Cell 57 (index 56): Histogramas D2 solo 3 variables ───
cells[56]['source'] = [
    "fig, axes = plt.subplots(1, len(vars_num_d2), figsize=(6*len(vars_num_d2), 5))\n"
    "if len(vars_num_d2) == 1: axes = [axes]\n"
    "\n"
    "for ax, col in zip(axes, vars_num_d2):\n"
    "    data = pd.to_numeric(df2[col], errors='coerce').dropna()\n"
    "    ax.hist(data, bins=25, color='#DD8452', edgecolor='white', alpha=0.85)\n"
    "    ax.axvline(data.mean(),   color='red',   linestyle='--', label=f'Media: {data.mean():.1f}')\n"
    "    ax.axvline(data.median(), color='green', linestyle='--', label=f'Mediana: {data.median():.1f}')\n"
    "    ax.set_title(titulos_d2.get(col, col), fontsize=9)\n"
    "    ax.set_xlabel('Valor')\n"
    "    ax.set_ylabel('Frecuencia')\n"
    "    ax.legend(fontsize=7)\n"
    "\n"
    "fig.suptitle('Histogramas - Dataset 2: Encuesta AF Completa', fontsize=14)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(OUT_DIR, 'histogramas_dataset2.png'), bbox_inches='tight', dpi=100)\n"
    "plt.show()\n"
]
cells[56]['outputs'] = []
cells[56]['execution_count'] = None
print("c56 OK")

# ─── Cell 61 (index 60): cat_d2 sin motivo_eliminacion ───
cells[60]['source'] = [
    "cat_d2 = [\n"
    "    ('sexo_rn_c',          'Sexo del RN'),\n"
    "    ('nivel_educacional_c', 'Nivel educacional materno'),\n"
    "    ('region_c',           'Region de Chile'),\n"
    "    ('primipara_c',        'Primipara'),\n"
    "    ('pandemia_c',         'Periodo pandemia COVID-19'),\n"
    "    ('categoria_imc',      'Categoria IMC pregestacional'),\n"
    "]\n"
    "cat_d2 = [(c, l) for c, l in cat_d2 if c in df2.columns]\n"
    "\n"
    "tablas_d2 = {}\n"
    "for col, label in cat_d2:\n"
    "    t = tabla_frecuencias(df2, col, label)\n"
    "    tablas_d2[col] = t\n"
    "    print(f'\\n--- {label} ---')\n"
    "    display(t)\n"
]
cells[60]['outputs'] = []
cells[60]['execution_count'] = None
print("c60 OK")

# ─── Cell 67 (index 66): Pie binarias con mapeo robusto ───
cells[66]['source'] = [
    "binarias = [\n"
    "    (df1, 'consumio_saf_binario', 'Consumio SAF - D1', {0: 'No consumio', 1: 'Si consumio'}),\n"
    "    (df1, 'sexo',       'Sexo del hijo - D1',  {1: 'Masculino', 2: 'Femenino'}),\n"
    "    (df2, 'pandemia_c', 'Pandemia - D2',        {0: 'Pre-pandemia', 1: 'Pandemia'}),\n"
    "    (df2, 'primipara_c','Primipara - D2',       {0: 'Multipara', 1: 'Primipara'}),\n"
    "    (df2, 'sexo_rn_c',  'Sexo RN - D2',         {1: 'Masculino', 2: 'Femenino'}),\n"
    "]\n"
    "binarias = [(d, c, t, m) for d, c, t, m in binarias if c in d.columns]\n"
    "\n"
    "fig, axes = plt.subplots(1, len(binarias), figsize=(4.5*len(binarias), 4))\n"
    "if len(binarias) == 1: axes = [axes]\n"
    "\n"
    "for ax, (df_src, col, titulo, mapa) in zip(axes, binarias):\n"
    "    s = pd.to_numeric(df_src[col], errors='coerce').dropna()\n"
    "    s = s[s.isin(mapa.keys())]\n"
    "    freq = s.map(mapa).value_counts(dropna=True)\n"
    "    if freq.empty:\n"
    "        ax.set_visible(False)\n"
    "        continue\n"
    "    ax.pie(freq.values, labels=freq.index, autopct='%1.1f%%',\n"
    "           colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(freq)],\n"
    "           startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})\n"
    "    ax.set_title(titulo, fontsize=10)\n"
    "\n"
    "fig.suptitle('Variables binarias - proporcion de categorias', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(OUT_DIR, 'pastel_binarias.png'), bbox_inches='tight', dpi=100)\n"
    "plt.show()\n"
]
cells[66]['outputs'] = []
cells[66]['execution_count'] = None
print("c66 OK")

# clear dependent cells outputs
for idx in [61, 62, 63, 64, 65, 67]:
    if idx < len(cells):
        cells[idx]['outputs'] = []
        cells[idx]['execution_count'] = None

with open('notebooks/etapa2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved. Total cells:", len(nb['cells']))
