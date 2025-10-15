# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Curva Elástica Interactiva", layout="wide")

# ---------- Utilidades simples (sin SciPy) ----------
def cumtrapz(y, x):
    """Integración acumulada por trapecios (y[0] = 0)."""
    dx = np.diff(x)
    area = (y[:-1] + y[1:]) * 0.5 * dx
    out = np.zeros_like(x, dtype=float)
    out[1:] = np.cumsum(area)
    return out

def max_with_index(x, y):
    i = int(np.argmax(y))
    return x[i], y[i]

def min_with_index(x, y):
    i = int(np.argmin(y))
    return x[i], y[i]

# ---------- Entrada de datos ----------
st.title("Proyecto Final – Curva Elástica Interactiva")

with st.sidebar:
    st.header("Datos de la viga")
    case = st.selectbox(
        "Caso de viga/carga",
        [
            "1) Simplemente apoyada + carga puntual en a",
            "2) Simplemente apoyada + carga distribuida uniforme (w)",
            "3) Voladizo + carga puntual en extremo",
            "4) Voladizo + carga distribuida uniforme (w)",
            "5) Voladizo + carga puntual a distancia a",
            "6) Simplemente apoyada + dos cargas puntuales (P1 en a1, P2 en a2)",
        ],
        index=0,
    )

    L = st.number_input("Longitud L [m]", min_value=0.1, value=5.0, step=0.1)
    E = st.number_input("Módulo de Young E [MPa] (ej. acero ~200000)", min_value=1.0, value=200000.0, step=1000.0)
    I = st.number_input("Inercia I [m^4] (ej. 8e-6)", min_value=1e-12, value=8e-6, step=1e-6, format="%.6e")

    # Parámetros de carga
    if case == "1) Simplemente apoyada + carga puntual en a":
        P = st.number_input("Carga P [kN]", value=10.0, step=0.5)
        a = st.number_input("Posición a [m] (desde el apoyo izquierdo)", min_value=0.0, max_value=L, value=L/2, step=0.1)
    elif case == "2) Simplemente apoyada + carga distribuida uniforme (w)":
        w = st.number_input("Carga distribuida w [kN/m]", value=2.0, step=0.1)
    elif case == "3) Voladizo + carga puntual en extremo":
        P = st.number_input("Carga P [kN]", value=10.0, step=0.5)
    elif case == "4) Voladizo + carga distribuida uniforme (w)":
        w = st.number_input("Carga distribuida w [kN/m]", value=2.0, step=0.1)
    elif case == "5) Voladizo + carga puntual a distancia a":
        P = st.number_input("Carga P [kN]", value=10.0, step=0.5)
        a = st.number_input("Posición a [m] (desde el empotramiento)", min_value=0.0, max_value=L, value=L/2, step=0.1)
    elif case == "6) Simplemente apoyada + dos cargas puntuales (P1 en a1, P2 en a2)":
        P1 = st.number_input("Carga P1 [kN]", value=8.0, step=0.5)
        a1 = st.number_input("Posición a1 [m]", min_value=0.0, max_value=L, value=L/3, step=0.1)
        P2 = st.number_input("Carga P2 [kN]", value=12.0, step=0.5)
        a2 = st.number_input("Posición a2 [m]", min_value=0.0, max_value=L, value=2*L/3, step=0.1)
        if a2 < a1:
            st.info("Nota: a2 debe ser ≥ a1 para la gráfica escalonada; puedes intercambiarlas si gustas.")

# Unidades internas (N, m) para E*I; kN -> N
kN = 1000.0
MPa = 1e6
E_SI = E * MPa  # [Pa] = N/m^2
EI = E_SI * I   # [N·m^2]

# Mallado
n = 400
x = np.linspace(0, L, n)

# ---------- Reacciones, V(x), M(x) y w(x) por caso ----------
w_x = np.zeros_like(x)
V = np.zeros_like(x)
M = np.zeros_like(x)
R_left = 0.0
R_right = 0.0
M_base = None  # momento en empotramiento (para voladizo)

if case == "1) Simplemente apoyada + carga puntual en a":
    P_N = P * kN
    # Reacciones (estática)
    R_left = P_N * (L - a) / L
    R_right = P_N * a / L

    # Carga (mostramos un "pincho" visual)
    w_x[:] = 0.0

    # Cortante y momento
    V[:] = R_left
    V[x >= a] = R_left - P_N
    M = R_left * x
    M[x >= a] = R_left * x[x >= a] - P_N * (x[x >= a] - a)

elif case == "2) Simplemente apoyada + carga distribuida uniforme (w)":
    w_Npm = w * kN  # N/m
    # Reacciones iguales
    R_left = R_right = w_Npm * L / 2
    w_x[:] = w_Npm

    V = R_left - w_Npm * x
    M = R_left * x - 0.5 * w_Npm * x**2

elif case == "3) Voladizo + carga puntual en extremo":
    P_N = P * kN
    # Envolvente: reacción vertical = P, momento en empotramiento = P*L
    M_base = P_N * L
    w_x[:] = 0.0

    V = -P_N * np.ones_like(x)  # negativo hacia abajo en todo el tramo
    M = -P_N * (L - x)          # máximo en x=0 (empotramiento)

elif case == "4) Voladizo + carga distribuida uniforme (w)":
    w_Npm = w * kN
    M_base = w_Npm * L**2 / 2
    w_x[:] = w_Npm

    V = -w_Npm * (L - x)
    M = -0.5 * w_Npm * (L - x)**2

elif case == "5) Voladizo + carga puntual a distancia a":
    P_N = P * kN
    M_base = P_N * (L - a)
    w_x[:] = 0.0

    V = np.zeros_like(x)
    V[x < a] = 0.0
    V[x >= a] = -P_N
    M = np.zeros_like(x)
    # tramo 0–a: momento constante por equilibrio de extremo libre (0)
    # tramo a–L: M(x) = -P*(x-a)
    M[x >= a] = -P_N * (x[x >= a] - a)

elif case == "6) Simplemente apoyada + dos cargas puntuales (P1 en a1, P2 en a2)":
    P1_N, P2_N = P1*kN, P2*kN
    # Reacciones
    R_left = (P1_N*(L - a1) + P2_N*(L - a2)) / L
    R_right = P1_N + P2_N - R_left
    w_x[:] = 0.0

    V[:] = R_left
    V[x >= a1] -= P1_N
    V[x >= a2] -= P2_N

    M = R_left * x
    M[x >= a1] -= P1_N * (x[x >= a1] - a1)
    M[x >= a2] -= P2_N * (x[x >= a2] - a2)

# ---------- Curva elástica por integración numérica ----------
kappa = M / EI  # curvatura = y''

theta_raw = cumtrapz(kappa, x)          # integración 1
y_raw = cumtrapz(theta_raw, x)          # integración 2

if case.startswith("1)") or case.startswith("2)") or case.startswith("6)"):
    # Simplemente apoyada: y(0)=0, y(L)=0  -> sumo línea C1*x + C2
    C2 = 0.0  # ya cumple en x=0
    C1 = -y_raw[-1] / L
    theta = theta_raw + C1
    y = y_raw + C1 * x + C2
else:
    # Voladizo: y(0)=0 y theta(0)=0 -> ya cumple por construcción
    theta = theta_raw
    y = y_raw

# ---------- Resultados máximos ----------
x_ymax, y_max = max_with_index(x, y)
x_ymin, y_min = min_with_index(x, y)
x_thmax, th_max = max_with_index(x, theta)
x_thmin, th_min = min_with_index(x, theta)

# ---------- Presentación ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Diagrama de cargas")
    fig, ax = plt.subplots()
    ax.plot(x, w_x/1000.0, linewidth=2)  # N/m -> kN/m
    # Flechas para cargas puntuales según el caso
    if case.startswith("1)"):
        ax.annotate("", xy=(a, max(0, np.max(w_x)/1000.0)+0.05),
                    xytext=(a, -max(0.2, np.max(w_x)/1000.0)+0.05),
                    arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(a, 0.02, f"P={P:.2f} kN", ha="center", va="bottom", rotation=90)
    if case.startswith("3)"):
        ax.annotate("", xy=(L, 0.2), xytext=(L, -0.8), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(L, 0.22, f"P={P:.2f} kN", ha="center")
    if case.startswith("5)"):
        ax.annotate("", xy=(a, 0.2), xytext=(a, -0.8), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(a, 0.22, f"P={P:.2f} kN", ha="center")
    if case.startswith("6)"):
        for (aa, PP) in [(a1, P1), (a2, P2)]:
            ax.annotate("", xy=(aa, 0.2), xytext=(aa, -0.8), arrowprops=dict(arrowstyle="->", lw=2))
            ax.text(aa, 0.22, f"{PP:.1f} kN", ha="center")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("w(x) [kN/m]")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Reacciones")
    if case.startswith(("1)", "2)", "6)")):
        st.write(f"**R_izq = {R_left/1000:.3f} kN**, **R_der = {R_right/1000:.3f} kN**")
    else:
        st.write(f"**Reacción vertical en empotramiento = {(-V[0])/1000:.3f} kN**")
        st.write(f"**Momento en empotramiento = {abs(M[0])/1000:.3f} kN·m**")

with col2:
    st.subheader("Diagrama de cortante V(x)")
    fig1, ax1 = plt.subplots()
    ax1.plot(x, V/1000.0, linewidth=2)  # N -> kN
    ax1.axhline(0, color="k", linewidth=1)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("V(x) [kN]")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    st.subheader("Diagrama de momento M(x)")
    fig2, ax2 = plt.subplots()
    ax2.plot(x, M/1000.0, linewidth=2)  # N·m -> kN·m
    ax2.axhline(0, color="k", linewidth=1)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("M(x) [kN·m]")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

st.subheader("Curva elástica y(x) y giro θ(x)")
col3, col4 = st.columns(2)
with col3:
    fig3, ax3 = plt.subplots()
    ax3.plot(x, y*1000, linewidth=2)  # m -> mm
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y(x) [mm]")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots()
    ax4.plot(x, theta, linewidth=2)
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("θ(x) [rad]")
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

# ---------- Valores máximos ----------
st.subheader("Valores máximos")
c1, c2 = st.columns(2)
with c1:
    st.write(f"**y_max** = {y_max*1000:.3f} mm en x = {x_ymax:.3f} m")
    st.write(f"**y_min** = {y_min*1000:.3f} mm en x = {x_ymin:.3f} m")
with c2:
    st.write(f"**θ_max** = {th_max:.6f} rad en x = {x_thmax:.3f} m")
    st.write(f"**θ_min** = {th_min:.6f} rad en x = {x_thmin:.3f} m")

# ---------- Manual breve ----------
with st.expander("Manual breve de uso"):
    st.markdown(
        """
**1. Selecciona el caso** en la barra lateral.  
**2. Ingresa L, E e I** (si no sabes I exacta, usa un valor de referencia).  
**3. Ingresa las cargas** (P, w, posiciones).  
**4. Observa:**  
- **Diagrama de cargas**, **reacciones**.  
- **V(x)**, **M(x)**.  
- **Curva elástica y(x)** (en **mm**) y **giro θ(x)** (en **rad**).  
- **Máximos** de deflexión y giro.  

**Notas:**  
- Signo positivo/negativo sigue la convención típica (cortante hacia arriba en apoyo izquierdo).  
- El cálculo de y(x) se hace por **integración numérica** de \( M/(EI) \).  
- Para **vigas simplemente apoyadas** se impone \(y(0)=y(L)=0\).  
- Para **voladizos** se impone \(y(0)=0\) y \(\\theta(0)=0\).
- Puedes aumentar n en el código si quieres malla más fina.
        """
    )

st.caption("Hecho para fines didácticos (Ingeniería Civil). Extensible para más casos agregando fórmulas de V y M.")
