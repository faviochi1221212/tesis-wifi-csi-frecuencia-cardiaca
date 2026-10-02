"""
Senal agregada y features por ventana.

Copias literales:
  senal_agregada, bpm_parabolico <- auditoria_fase_A/A1b_validar_reproduccion.py L25-45 (copias de 12 L127-148)
  FEAT_NAMES, extraer_features_ventana <- auditoria_fase_A/A2_calidad_csi.py L47-49, L229-275
                                          (copia de 12_features_modelos_v3_synced.py L181-254; la usa E1_rf)

Las 10 features de E1a (config.HISTORICAL_FEATURES_E1A) son HISTORICAS: N=10 se eligio mirando un test en
23_seleccion_mi.py. Aqui no se hace ninguna seleccion.
"""
import numpy as np

from .config import BUTTER_HIGH_HZ, BUTTER_LOW_HZ, HISTORICAL_FEATURES_E1A  # noqa: F401  (re-export)

FEAT_NAMES = ["media", "std", "varianza", "energia", "rms", "pico_pico", "e_banda", "ratio_e", "freq_dom", "bpm_est",
              "centroide", "ancho_banda", "pico_ac", "bpm_ac", "wav_e0", "wav_e1", "wav_e2", "wav_e3", "entropia",
              "bpm_harm", "bpm_ac_parab", "snr_cardiac", "kurtosis_s", "skewness_s"]


def senal_agregada(datos, indices):
    segs = []
    for i in indices:
        s = datos[:, i].astype(float); rng = s.max() - s.min()
        if rng > 1e-10:
            s = (s - s.mean()) / rng
        segs.append(s)
    return np.mean(segs, axis=0)


def bpm_parabolico(freqs, fft_mag, mask):
    freqs_b = freqs[mask]; fft_b = fft_mag[mask]
    if len(fft_b) == 0: return 0.0
    idx = int(np.argmax(fft_b))
    if 0 < idx < len(fft_b) - 1:
        y0, y1, y2 = fft_b[idx-1], fft_b[idx], fft_b[idx+1]
        denom = y0 - 2*y1 + y2
        delta = 0.5*(y0 - y2) / (denom + 1e-10) if abs(denom) > 1e-10 else 0.0
        df = freqs_b[1] - freqs_b[0] if len(freqs_b) > 1 else 0.0
        return (freqs_b[idx] + delta * df) * 60.0
    return freqs_b[idx] * 60.0


def extraer_features_ventana(senal, fs):
    """Copia literal de 12_features_modelos_v3_synced.py L181-254 (sin cambios de logica)."""
    import pywt
    from scipy.stats import kurtosis as sk, skew as ssk
    F_LOW, F_HIGH = BUTTER_LOW_HZ, BUTTER_HIGH_HZ
    n = len(senal)
    media = float(np.mean(senal)); std = float(np.std(senal)); varianza = float(np.var(senal))
    energia = float(np.sum(senal ** 2)); rms = float(np.sqrt(np.mean(senal ** 2))); pico_pico = float(senal.max() - senal.min())
    kurt = float(sk(senal)); skewn = float(ssk(senal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs); fft_mag = np.abs(np.fft.rfft(senal * np.hanning(n)))
    mask = (freqs >= F_LOW) & (freqs <= F_HIGH)
    e_banda = float(np.sum(fft_mag[mask] ** 2)); e_total = float(np.sum(fft_mag ** 2)) + 1e-10
    ratio_e = e_banda / e_total; snr_db = float(10.0 * np.log10(e_banda / (e_total - e_banda + 1e-10)))
    bpm_fft_par = bpm_parabolico(freqs, fft_mag, mask); freq_dom = bpm_fft_par / 60.0
    cand = freqs[mask]; scores = np.zeros(len(cand))
    for k, f0 in enumerate(cand):
        i0 = np.argmin(np.abs(freqs - f0)); sc = fft_mag[i0]
        if 2 * f0 <= freqs[-1]:
            sc = sc + 0.3 * fft_mag[np.argmin(np.abs(freqs - 2 * f0))]
        scores[k] = sc
    bpm_harm = cand[int(np.argmax(scores))] * 60.0
    pesos = fft_mag[mask]; centroide = float(np.sum(freqs[mask] * pesos) / (np.sum(pesos) + 1e-10))
    umbral = fft_mag[mask].max() / np.sqrt(2); den = np.sum(fft_mag[mask] >= umbral) + 1e-10
    ancho = float(np.sum(freqs[mask][fft_mag[mask] >= umbral]) / den)
    sc_ = senal - np.mean(senal); ac = np.correlate(sc_, sc_, mode='full'); ac = ac[len(ac) // 2:]; ac = ac / (ac[0] + 1e-10)
    lag_min = max(1, int(fs / F_HIGH)); lag_max = min(n - 1, int(fs / F_LOW))
    if lag_max > lag_min and lag_max < len(ac):
        seg = ac[lag_min:lag_max]; pico_ac = float(np.max(seg)); lag_p = int(np.argmax(seg)) + lag_min
        bpm_ac = float((fs / (lag_p + 1e-10)) * 60.0)
        pk = int(np.argmax(seg))
        if 0 < pk < len(seg) - 1:
            y0, y1, y2 = seg[pk - 1], seg[pk], seg[pk + 1]; dn = y0 - 2 * y1 + y2
            dl = 0.5 * (y0 - y2) / (dn + 1e-10) if abs(dn) > 1e-10 else 0.0
            bpm_ac_p = (fs / ((pk + dl) + lag_min + 1e-10)) * 60.0
        else:
            bpm_ac_p = (fs / (float(pk + lag_min) + 1e-10)) * 60.0
    else:
        pico_ac = bpm_ac = bpm_ac_p = 0.0
    try:
        coeffs = pywt.wavedec(senal, 'db4', level=3); e_wav = [float(np.sum(c ** 2)) for c in coeffs]
        tot = sum(e_wav) + 1e-10; rw = [e / tot for e in e_wav]
        d1 = coeffs[-1]; d1n = d1 ** 2 / (np.sum(d1 ** 2) + 1e-10); entropia = float(-np.sum(d1n * np.log(d1n + 1e-10)))
    except Exception:
        rw = [0.0] * 4; entropia = 0.0
    return dict(zip(FEAT_NAMES, [media, std, varianza, energia, rms, pico_pico, e_banda, ratio_e, freq_dom, bpm_fft_par,
                                 centroide, ancho, pico_ac, bpm_ac, rw[0], rw[1], rw[2], rw[3], entropia, bpm_harm, bpm_ac_p,
                                 snr_db, kurt, skewn]))
