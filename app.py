"""TMEM120A molecular screening webtool using the final RBF-SVR backend."""
from __future__ import annotations

import html
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, FilterCatalog, Lipinski, QED, rdMolDescriptors
from streamlit_ketcher import st_ketcher

from predictor_svr import predict_docking_score

st.set_page_config(
    page_title="TMEM120A Molecular Screening Platform",
    page_icon="🧬",
    layout="wide",
)


# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    button, input, textarea, select, label, p, li, table {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        letter-spacing: -0.025em;
        font-weight: 700;
    }
    code, pre, .statusline, .model-flow, .pstep .num, .pstep .state {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
    }
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1450px; }
    .hero {
        border: 1px solid rgba(56,189,248,.22);
        border-radius: 18px;
        padding: 1.45rem 1.6rem 1.25rem 1.6rem;
        background: linear-gradient(135deg, rgba(14,165,233,.09), rgba(99,102,241,.05));
        margin-bottom: 1.2rem;
    }
    .hero-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.035em; margin-bottom: .2rem; }
    .hero-sub { color: #8b95a7; font-size: 1rem; line-height: 1.55; margin-bottom: .85rem; }
    .badges { display: flex; gap: .45rem; flex-wrap: wrap; }
    .badge {
        display: inline-block; padding: .27rem .58rem; border-radius: 999px;
        border: 1px solid rgba(56,189,248,.28); color: #7dd3fc;
        background: rgba(14,165,233,.07); font-size: .78rem; font-weight: 650;
        letter-spacing: .02em;
    }
    .statusline { margin-top:.8rem; color:#94a3b8; font-size:.82rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .status-dot { color:#34d399; text-shadow:0 0 10px rgba(52,211,153,.75); }
    .section-card {
        border: 1px solid rgba(148,163,184,.15); border-radius: 14px;
        padding: 1rem 1.1rem; background: rgba(15,23,42,.025);
    }
    .pipeline-wrap {
        display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:.58rem;
        margin:.65rem 0 1rem 0;
    }
    .pstep {
        min-height:92px; border-radius:12px; padding:.72rem .72rem .65rem .72rem;
        border:1px solid rgba(148,163,184,.20); background:rgba(100,116,139,.045);
        transition:all .25s ease; position:relative; overflow:hidden;
    }
    .pstep .num { font-family:ui-monospace,monospace; font-size:.72rem; color:#64748b; }
    .pstep .name { margin-top:.28rem; font-size:.82rem; font-weight:680; line-height:1.15rem; }
    .pstep .state { margin-top:.46rem; font-family:ui-monospace,monospace; font-size:.66rem; letter-spacing:.07em; }
    .waiting { opacity:.42; }
    .waiting .state { color:#64748b; }
    .processing {
        border-color:#22d3ee; background:rgba(34,211,238,.075);
        box-shadow:0 0 0 1px rgba(34,211,238,.13), 0 0 22px rgba(34,211,238,.24);
        animation:pulseGlow 1s ease-in-out infinite alternate;
    }
    .processing .state { color:#22d3ee; }
    .done {
        border-color:rgba(52,211,153,.58); background:rgba(52,211,153,.055);
        box-shadow:0 0 12px rgba(52,211,153,.09);
    }
    .done .state { color:#34d399; }
    .error { border-color:#fb7185; background:rgba(251,113,133,.06); }
    .error .state { color:#fb7185; }
    @keyframes pulseGlow {
        from { box-shadow:0 0 8px rgba(34,211,238,.16); }
        to   { box-shadow:0 0 26px rgba(34,211,238,.40); }
    }
    .model-flow {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size:.84rem; line-height:1.65rem; padding:.75rem .9rem;
        border-radius:10px; background:rgba(15,23,42,.04);
        border:1px solid rgba(148,163,184,.16);
    }
    @media (max-width: 1000px) { .pipeline-wrap { grid-template-columns:repeat(3,1fr); } }
    @media (max-width: 640px) { .pipeline-wrap { grid-template-columns:repeat(2,1fr); } }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# MOLECULAR UTILITIES
# -----------------------------------------------------------------------------
@st.cache_resource
def alert_catalogs():
    pains_params = FilterCatalog.FilterCatalogParams()
    for catalog in (
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    ):
        pains_params.AddCatalog(catalog)
    brenk_params = FilterCatalog.FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(pains_params), FilterCatalog.FilterCatalog(brenk_params)


def make_3d_block(mol: Chem.Mol) -> str:
    mol_3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol_3d, params)
    if status != 0:
        raise ValueError("Could not generate a 3D conformer for this molecule.")
    AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
    return Chem.MolToMolBlock(mol_3d)


def render_3d(mol_block: str, height: int = 680) -> None:
    escaped = mol_block.replace("`", "\\`").replace("${", "\\${")
    viewer_html = f"""
    <div id="viewer" style="height:{height - 20}px;width:100%;position:relative;border-radius:14px;overflow:hidden;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const viewer = $3Dmol.createViewer('viewer', {{backgroundColor: '#0f172a'}});
      viewer.addModel(`{escaped}`, 'sdf');
      viewer.setStyle({{}}, {{stick: {{radius: 0.16}}, sphere: {{scale: 0.28}}}});
      viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.18, color: 'white'}});
      viewer.zoomTo(); viewer.render(); viewer.zoom(1.15, 0);
    </script>
    """
    components.html(viewer_html, height=height)


def properties(mol: Chem.Mol) -> dict[str, float | int | str]:
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight": Descriptors.MolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "TPSA (Å²)": rdMolDescriptors.CalcTPSA(mol),
        "H-bond donors": Lipinski.NumHDonors(mol),
        "H-bond acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable bonds": Lipinski.NumRotatableBonds(mol),
        "Rings": Lipinski.RingCount(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Molar refractivity": Crippen.MolMR(mol),
        "Fraction Csp³": rdMolDescriptors.CalcFractionCSP3(mol),
        "QED": QED.qed(mol),
    }


def filter_results(mol: Chem.Mol, p: dict[str, float | int | str]) -> list[dict[str, str]]:
    def result(name: str, passed: bool, criteria: str, note: str = "") -> dict[str, str]:
        return {
            "Filter": name,
            "Result": "Pass" if passed else "Flag",
            "Criteria": criteria,
            "Details": note or ("Within guideline" if passed else "Outside guideline"),
        }

    lipinski_violations = sum((
        p["Molecular weight"] > 500,
        p["cLogP"] > 5,
        p["H-bond donors"] > 5,
        p["H-bond acceptors"] > 10,
    ))
    pains_catalog, brenk_catalog = alert_catalogs()
    pains = [match.GetDescription() for match in pains_catalog.GetMatches(mol)]
    brenk = [match.GetDescription() for match in brenk_catalog.GetMatches(mol)]

    return [
        result("Lipinski Ro5", lipinski_violations <= 1, "≤1 violation: MW ≤500; cLogP ≤5; HBD ≤5; HBA ≤10", f"{lipinski_violations} violation(s)"),
        result("Veber", p["Rotatable bonds"] <= 10 and p["TPSA (Å²)"] <= 140, "Rotatable bonds ≤10; TPSA ≤140 Å²"),
        result("Egan", p["cLogP"] <= 5.88 and p["TPSA (Å²)"] <= 131.6, "cLogP ≤5.88; TPSA ≤131.6 Å²"),
        result("Ghose", 160 <= p["Molecular weight"] <= 480 and -0.4 <= p["cLogP"] <= 5.6 and 40 <= p["Molar refractivity"] <= 130 and 20 <= p["Heavy atoms"] <= 70, "MW 160–480; cLogP −0.4–5.6; MR 40–130; atoms 20–70"),
        result("Muegge", 200 <= p["Molecular weight"] <= 600 and -2 <= p["cLogP"] <= 5 and p["TPSA (Å²)"] <= 150 and p["H-bond donors"] <= 5 and p["H-bond acceptors"] <= 10 and p["Rotatable bonds"] <= 15 and p["Rings"] <= 7, "MW 200–600; cLogP −2–5; TPSA ≤150; HBD ≤5; HBA ≤10; RB ≤15; rings ≤7"),
        result("PAINS alerts", not pains, "No PAINS substructure alerts", "; ".join(pains) if pains else "No PAINS alert"),
        result("Brenk alerts", not brenk, "No Brenk structural alerts", "; ".join(brenk) if brenk else "No Brenk alert"),
    ]


def property_table(p: dict[str, float | int | str]) -> pd.DataFrame:
    rows = []
    for name, value in p.items():
        value_text = f"{value:.2f}" if isinstance(value, float) else str(value)
        rows.append({"Property": name, "Value": value_text})
    return pd.DataFrame(rows)


def export_table(smiles: str, target_name: str, prediction, p, filters) -> pd.DataFrame:
    row: dict[str, str | float | int] = {
        "Input SMILES": smiles,
        "Protein target": target_name,
        "Predicted docking score (kcal/mol)": round(prediction.score, 3) if prediction.score is not None else "",
        "Predicted category": prediction.category if prediction.in_domain else "",
        "Model applicability": "IN_DOMAIN" if prediction.in_domain else "OUT_OF_DOMAIN",
        "Descriptor max |z|": round(prediction.max_abs_z, 3),
        "OOD guard threshold": round(prediction.ood_threshold, 3),
        "Nearest training Tanimoto": prediction.nearest_training_similarity,
        "Nearest training SMILES": prediction.nearest_training_smiles,
        "Nearest training docking score (kcal/mol)": prediction.nearest_training_docking_score,
        "Model": prediction.model_name,
    }
    row.update({f"Property: {name}": value for name, value in p.items()})
    for item in filters:
        row[f"Filter: {item['Filter']}"] = item["Result"]
        row[f"Details: {item['Filter']}"] = item["Details"]
    return pd.DataFrame([row])


def use_drawn_smiles(drawn_smiles: str) -> None:
    st.session_state["smiles_input"] = drawn_smiles


# -----------------------------------------------------------------------------
# PIPELINE VISUALIZATION
# -----------------------------------------------------------------------------
PIPELINE = [
    ("structure", "Structure parsing"),
    ("descriptors", "RDKit + Mordred"),
    ("features", "931-feature mapping"),
    ("standardize", "StandardScaler"),
    ("inference", "RBF-SVR inference"),
    ("report", "Report generation"),
]


def pipeline_html(states: dict[str, str]) -> str:
    cards = []
    for i, (key, label) in enumerate(PIPELINE, 1):
        state = states.get(key, "waiting")
        state_label = {"waiting": "WAITING", "processing": "PROCESSING", "done": "COMPLETE", "error": "ERROR"}[state]
        symbol = {"waiting": "○", "processing": "●", "done": "✓", "error": "!"}[state]
        cards.append(
            f'<div class="pstep {state}"><div class="num">0{i}</div>'
            f'<div class="name">{html.escape(label)}</div>'
            f'<div class="state">{symbol} {state_label}</div></div>'
        )
    return '<div class="pipeline-wrap">' + "".join(cards) + "</div>"


# -----------------------------------------------------------------------------
# HEADER + MODEL CARD
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">TMEM120A Molecular Screening Platform</div>
      <div class="hero-sub">AI-assisted docking-score estimation and molecular property analysis for rapid TMEM120A candidate prioritization.</div>
      <div class="badges">
        <span class="badge">RBF-SVR</span>
        <span class="badge">931 FEATURES</span>
        <span class="badge">RDKit + Mordred</span>
        <span class="badge">R² 0.6357</span>
        <span class="badge">RMSE 0.5752 kcal/mol</span>
      </div>
      <div class="statusline"><span class="status-dot">●</span> MODEL ONLINE &nbsp;&nbsp;|&nbsp;&nbsp; TMEM120A QSAR ENGINE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("About the RBF-SVR model"):
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown(
            "The model was developed from **809 compounds** with reference TMEM120A docking scores. "
            "RDKit and Mordred 2D descriptors were calculated, preprocessed, and reduced to **931 retained features**. "
            "The final predictor uses a **StandardScaler → radial-basis-function support vector regression (RBF-SVR)** pipeline."
        )
        st.markdown(
            '<div class="model-flow">SMILES → RDKit + Mordred → 1,526 descriptors → correlation filtering → 931 features → StandardScaler → RBF-SVR → predicted docking score</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            "**Final model**  \n"
            "C = 8 · γ = 0.0003 · ε = 0.40  \n"
            "Training = 607 · Independent test = 202  \n"
            "R² = 0.6357 · RMSE = 0.5752 kcal/mol  \n"
            "MAE = 0.4624 kcal/mol · Spearman = 0.6864"
        )
        st.caption("The predicted score is a QSAR-based estimate for prioritization, not a replacement for explicit docking or experimental validation.")

with st.expander("How to use this platform"):
    st.markdown(
        "**1. Input a molecule** — enter a SMILES string or draw a structure.  \n"
        "**2. Run analysis** — the platform calculates RDKit/Mordred descriptors and maps the retained 931 model features.  \n"
        "**3. Predict** — standardized features are evaluated by the final RBF-SVR model.  \n"
        "**4. Review** — inspect the docking-score estimate, applicability flag, 2D/3D structure, physicochemical profile, drug-likeness filters, and nearest training analogue.  \n"
        "**5. Export** — download the molecular analysis as CSV."
    )


target_name = "TMEM120A"
with st.sidebar:
    st.markdown("**Example SMILES**")
    st.code("CC(=O)Nc1ccc(O)cc1", language=None)
    st.caption("Paracetamol")
    st.code("CCN(CC)CC(=O)Nc1c(C)cccc1C", language=None)
    st.caption("Lidocaine")
    st.code("CC(C)c1cccc(C(C)C)c1O", language=None)
    st.caption("Propofol")

smiles_tab, draw_tab = st.tabs(["Enter SMILES", "Draw molecule"])
with smiles_tab:
    smiles = st.text_area(
        "Input SMILES",
        value="CC(=O)Nc1ccc(O)cc1",
        height=100,
        placeholder="e.g. CC(=O)Nc1ccc(O)cc1",
        key="smiles_input",
    )
with draw_tab:
    st.caption("Draw or edit a molecule below, then transfer its generated SMILES into the predictor.")
    drawn_smiles = st_ketcher(st.session_state.get("smiles_input", "CC(=O)Nc1ccc(O)cc1"))
    if drawn_smiles:
        if Chem.MolFromSmiles(drawn_smiles) is None:
            st.warning("The drawn structure does not yet produce a valid SMILES. Please complete the structure.")
        else:
            st.code(drawn_smiles, language=None)
            st.button("Use drawn structure for prediction", type="primary", on_click=use_drawn_smiles, args=(drawn_smiles,))
    else:
        st.info("Finish drawing a molecule to generate its SMILES.")

pipeline_placeholder = st.empty()
pipeline_placeholder.markdown(pipeline_html({}), unsafe_allow_html=True)

if st.button("Run molecular analysis", type="primary", use_container_width=True):
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        st.error("Invalid SMILES. Please check the molecular notation and try again.")
        st.stop()

    states = {key: "waiting" for key, _ in PIPELINE}

    def stage_callback(stage: str, state: str) -> None:
        states[stage] = state
        pipeline_placeholder.markdown(pipeline_html(states), unsafe_allow_html=True)

    try:
        # 3D and physicochemical outputs are prepared in parallel with the scientific report stage.
        prediction = predict_docking_score(mol, target_name, stage_callback=stage_callback)
        mol_block = make_3d_block(mol)
        props = properties(mol)
        filters = filter_results(mol, props)
    except Exception as exc:
        active = next((k for k, v in states.items() if v == "processing"), None)
        if active:
            states[active] = "error"
        pipeline_placeholder.markdown(pipeline_html(states), unsafe_allow_html=True)
        st.error("The RBF-SVR molecular analysis backend could not run.")
        st.exception(exc)
        st.stop()

    lipinski = next(item for item in filters if item["Filter"] == "Lipinski Ro5")
    st.success("ANALYSIS COMPLETE")

    score_col, category_col, qed_col, filter_col = st.columns(4)
    if prediction.in_domain:
        score_col.metric("Predicted docking score", f"{prediction.score:.2f} kcal/mol", help="QSAR-estimated docking score. More negative values correspond to stronger-score categories used in this project.")
        category_col.metric("Predicted category", prediction.category)
    else:
        score_col.metric("Predicted docking score", "Not reported", help="The molecule lies beyond the empirical 931-feature guard calibrated from the training representation.")
        category_col.metric("Model applicability", "OUT OF DOMAIN")
        st.warning(
            "Docking score/category were not reported because this molecule lies far outside the 931-feature space represented by the training set. "
            "This guard is a conservative feature-space check, not a calibrated prediction-confidence probability."
        )
    qed_col.metric("QED drug-likeness", f"{props['QED']:.2f}")
    filter_col.metric("Lipinski rule-of-five", lipinski["Result"], help=lipinski["Details"])

    st.caption(
        f"Target: **{target_name}** · Descriptor max |z|: **{prediction.max_abs_z:.2f}** · "
        f"OOD guard threshold: **{prediction.ood_threshold:.2f}** · Model: **RBF-SVR**"
    )

    with st.expander("Model context and nearest training analogue"):
        analogue_info, analogue_structure = st.columns([1.15, 1], gap="large")
        with analogue_info:
            st.markdown("**Nearest compound in the training set**")
            st.write(f"Morgan Tanimoto similarity: **{prediction.nearest_training_similarity:.3f}**")
            st.write(f"Reference docking score: **{prediction.nearest_training_docking_score:.3f} kcal/mol**")
            st.markdown("**Training SMILES**")
            st.code(prediction.nearest_training_smiles, language=None)
            st.caption("Structural similarity is provided as context only and is not a calibrated confidence score.")
        with analogue_structure:
            st.markdown("**2D structure of nearest training analogue**")
            analogue_mol = Chem.MolFromSmiles(prediction.nearest_training_smiles or "")
            if analogue_mol is not None:
                AllChem.Compute2DCoords(analogue_mol)
                st.image(
                    Draw.MolToImage(analogue_mol, size=(520, 320)),
                    use_container_width=True,
                )
            else:
                st.info("A valid 2D structure could not be generated for the nearest training analogue.")

    viewer_col, profile_col = st.columns([1.55, 1], gap="large")
    with viewer_col:
        st.subheader("3D structure")
        render_3d(mol_block, height=680)
    with profile_col:
        st.subheader("Molecular profile")
        mol_2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_2d)
        st.image(Draw.MolToImage(mol_2d, size=(600, 330)), use_container_width=True)
        st.markdown("**Physicochemical properties**")
        st.dataframe(property_table(props), use_container_width=True, hide_index=True, height=390)

    st.subheader("Drug-likeness and structural-alert filters")
    filter_df = pd.DataFrame(filters)
    st.dataframe(
        filter_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Result": st.column_config.TextColumn("Result", width="small"),
            "Criteria": st.column_config.TextColumn("Criteria", width="large"),
            "Details": st.column_config.TextColumn("Details", width="medium"),
        },
    )
    st.caption("These are medicinal-chemistry screening heuristics. A flag does not prove a compound is unsuitable; it indicates an aspect worth reviewing.")

    export_df = export_table(smiles.strip(), target_name, prediction, props, filters)
    st.download_button(
        "Download results as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="TMEM120A_RBF_SVR_prediction.csv",
        mime="text/csv",
    )

    with st.expander("Canonical SMILES and model details"):
        st.code(prediction.canonical_smiles, language=None)
        st.markdown("**Model architecture:** RDKit + Mordred 2D descriptors → 931 retained features → StandardScaler → RBF-SVR.")
        st.markdown(
            f"**Independent test:** R² {prediction.model_test_r2:.4f} · RMSE {prediction.model_test_rmse:.4f} kcal/mol · "
            f"MAE {prediction.model_test_mae:.4f} kcal/mol · Pearson {prediction.model_test_pearson:.4f} · "
            f"Spearman {prediction.model_test_spearman:.4f}."
        )

st.divider()
st.markdown(
    "<div style='text-align:center; color:#8b95a7; padding:0.4rem 0 1rem;'>"
    "© Made by Duy · CBMC Lab · College of Pharmacy · Seoul National University"
    "</div>",
    unsafe_allow_html=True,
)
