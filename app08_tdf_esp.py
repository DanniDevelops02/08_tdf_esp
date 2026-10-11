import re

import pandas as pd
import streamlit as st
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Configuración de página ----------
st.set_page_config(
    page_title="TF-IDF en Español",
    page_icon="🔍",
    layout="centered",
)

# ---------- Estilo (misma paleta que app.py) ----------
PRIMARIO = "#3f71a6"
OSCURO = "#1b2f45"
CLARO = "#eaf2f9"
SUAVE = "#a9c7e3"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif;
        font-size: 1.08rem;
    }}
    .stApp {{
        background: linear-gradient(180deg, #f6fafd 0%, {CLARO} 100%);
    }}
    .block-container {{
        max-width: 860px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }}
    .hero {{
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }}
    .hero .saludo {{
        color: {PRIMARIO};
        letter-spacing: .25em;
        text-transform: uppercase;
        font-size: .85rem;
        font-weight: 600;
    }}
    .hero h1 {{
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        color: {OSCURO};
        margin: .3rem 0 .6rem 0;
        line-height: 1.1;
    }}
    .hero p {{
        color: #4d6479;
        font-size: 1.2rem;
        max-width: 640px;
        margin: 0 auto;
    }}
    .linea {{
        width: 70px;
        height: 4px;
        border-radius: 4px;
        background: {PRIMARIO};
        margin: 1.4rem auto;
    }}
    h2, h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {OSCURO} !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: white;
        border: 1px solid {SUAVE}66 !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 24px rgba(63, 113, 166, .08);
        padding: .6rem .8rem;
    }}
    .stButton > button {{
        background: {PRIMARIO};
        color: white;
        border: none;
        border-radius: 999px;
        padding: .7rem 2.2rem;
        font-size: 1.05rem;
        font-weight: 600;
        transition: all .2s ease;
    }}
    .stButton > button:hover {{
        background: {OSCURO};
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(27, 47, 69, .25);
    }}
    /* Botones de preguntas sugeridas: versión suave */
    [class*="st-key-sug_"] button {{
        background: {CLARO};
        color: {OSCURO};
        border: 1px solid {SUAVE};
        border-radius: 14px;
        padding: .5rem 1rem;
        font-size: .95rem;
        font-weight: 500;
        text-align: left;
    }}
    [class*="st-key-sug_"] button:hover {{
        background: {PRIMARIO};
        color: white;
    }}
    .stTextInput input, .stTextArea textarea {{
        border-radius: 12px;
        border: 1px solid {SUAVE};
        font-size: 1.05rem;
    }}
    [data-testid="stAlert"] {{
        border-radius: 14px;
    }}
    .respuesta {{
        background: {CLARO};
        border-left: 5px solid {PRIMARIO};
        border-radius: 14px;
        padding: 1rem 1.2rem;
        font-size: 1.15rem;
        color: {OSCURO};
        margin: .4rem 0 .8rem 0;
    }}
    [data-testid="stSidebar"] {{
        background: {OSCURO};
    }}
    [data-testid="stSidebar"] * {{
        color: #e6f0f9 !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Datos de ejemplo ----------
DOCUMENTOS = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

SUGERIDAS = [
    "¿Dónde juegan el perro y el gato?",
    "¿Qué hacen los niños en el parque?",
    "¿Cuándo cantan los pájaros?",
    "¿Dónde suena la música alta?",
    "¿Qué animal maúlla durante la noche?",
]

# Palabras vacías: aparecen en casi todas las frases y no aportan significado
PALABRAS_VACIAS = set("""
el la los las un una unos unas y o u e en de del al a por para con sin se su sus
que qué es son muy lo le les mi tu como cómo donde dónde cuando cuándo cual cuál
quien quién hace hacen
""".split())

stemmer = SnowballStemmer("spanish")

if "pregunta" not in st.session_state:
    st.session_state.pregunta = SUGERIDAS[0]


def usar_sugerida(texto):
    st.session_state.pregunta = texto


def tokenizar(texto, quitar_vacias):
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto.lower())
    tokens = [t for t in texto.split() if len(t) > 1]
    if quitar_vacias:
        tokens = [t for t in tokens if t not in PALABRAS_VACIAS]
    return [stemmer.stem(t) for t in tokens]


# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Ajustes")
    quitar_vacias = st.checkbox(
        "Ignorar palabras vacías",
        value=True,
        help="Quita palabras como 'el', 'en', 'los' que se repiten en todas las frases.",
    )
    umbral = st.slider("Similitud mínima para confiar", 0.0, 1.0, 0.10, 0.01)
    st.caption(
        "TF-IDF le da más peso a las palabras que son frecuentes en un documento "
        "pero raras en el resto. La similitud coseno compara la pregunta con cada documento."
    )

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <div class="saludo">Procesamiento de lenguaje</div>
        <h1>TF-IDF en Español</h1>
        <p>Escribe unos documentos, hazles una pregunta y la app encuentra
        la frase que mejor la responde.</p>
        <div class="linea"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Entrada ----------
col1, col2 = st.columns([3, 2], gap="medium")

with col1:
    with st.container(border=True):
        st.subheader("Documentos")
        texto_docs = st.text_area("Uno por línea", DOCUMENTOS, height=210)
        pregunta = st.text_input("Tu pregunta", key="pregunta")

with col2:
    with st.container(border=True):
        st.subheader("Preguntas sugeridas")
        for i, sugerida in enumerate(SUGERIDAS):
            st.button(
                sugerida,
                key=f"sug_{i}",
                on_click=usar_sugerida,
                args=(sugerida,),
                width="stretch",
            )

st.write("")
analizar = st.button("🔍 Analizar", type="primary")

# ---------- Análisis ----------
if analizar:
    documentos = [d.strip() for d in texto_docs.split("\n") if d.strip()]

    if not documentos:
        st.error("Ingresa al menos un documento.")
    elif not pregunta.strip():
        st.error("Escribe una pregunta.")
    else:
        vectorizador = TfidfVectorizer(
            tokenizer=lambda t: tokenizar(t, quitar_vacias),
            token_pattern=None,
            lowercase=False,
        )
        try:
            X = vectorizador.fit_transform(documentos)
        except ValueError:
            st.error("Los documentos no tienen palabras útiles para analizar.")
            st.stop()

        similitudes = cosine_similarity(vectorizador.transform([pregunta]), X).flatten()
        mejor = int(similitudes.argmax())
        puntaje = float(similitudes[mejor])

        st.write("")
        with st.container(border=True):
            st.subheader("Respuesta")
            st.caption(f"Tu pregunta: {pregunta}")
            if puntaje >= umbral:
                st.markdown(f'<div class="respuesta">{documentos[mejor]}</div>', unsafe_allow_html=True)
                st.success(f"Similitud: {puntaje:.3f}")
            elif puntaje > 0:
                st.markdown(f'<div class="respuesta">{documentos[mejor]}</div>', unsafe_allow_html=True)
                st.warning(f"Baja confianza · similitud: {puntaje:.3f}")
            else:
                st.warning("Ningún documento comparte palabras con tu pregunta.")

        st.write("")
        with st.container(border=True):
            st.subheader("Similitud con cada documento")
            ranking = pd.DataFrame(
                {"Documento": documentos, "Similitud": similitudes.round(3)},
                index=[f"Doc {i + 1}" for i in range(len(documentos))],
            ).sort_values("Similitud", ascending=False)
            st.dataframe(
                ranking,
                width="stretch",
                column_config={
                    "Similitud": st.column_config.ProgressColumn(
                        "Similitud", min_value=0.0, max_value=1.0, format="%.3f"
                    )
                },
            )

        st.write("")
        with st.container(border=True):
            st.subheader("Matriz TF-IDF")
            st.caption("Cada columna es la raíz de una palabra; cada valor, su peso en el documento.")
            matriz = pd.DataFrame(
                X.toarray(),
                columns=vectorizador.get_feature_names_out(),
                index=[f"Doc {i + 1}" for i in range(len(documentos))],
            )
            st.dataframe(matriz.round(3), width="stretch")
