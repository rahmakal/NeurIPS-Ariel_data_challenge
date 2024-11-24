import numpy as np
import pandas as pd
from scipy.optimize import minimize


def phase_detector(signal):
    MIN = np.argmin(signal[30:140]) + 30
    signal1 = signal[:MIN]
    signal2 = signal[MIN:]

    first_derivative1 = np.gradient(signal1)
    first_derivative1 /= first_derivative1.max()
    first_derivative2 = np.gradient(signal2)
    first_derivative2 /= first_derivative2.max()

    phase1 = np.argmin(first_derivative1)
    phase2 = np.argmax(first_derivative2) + MIN

    return phase1, phase2


def predict_spectra(signal):
    def objective_to_minimize(s):
        delta = 2
        power = 3
        x = list(range(signal.shape[0] - delta * 4))
        y = (
            signal[: phase1 - delta].tolist()
            + (signal[phase1 + delta : phase2 - delta] * (1 + s)).tolist()
            + signal[phase2 + delta :].tolist()
        )

        z = np.polyfit(x, y, deg=power)
        p = np.poly1d(z)
        q = np.abs(p(x) - y).mean()
        return q

    signal = signal[:, 1:].mean(axis=1)
    phase1, phase2 = phase_detector(signal)

    s = minimize(fun=objective_to_minimize, x0=[0.0001], method="Nelder-Mead").x[0]
    return s

def predict(preprocessed_signal):
    predictions_spectra = [
    predict_spectra(preprocessed_signal[i]) for i in range(len(preprocessed_signal))
    ]
    return predictions_spectra

def submit(predictions_spectra):
    wave_to_apriori_scale = pd.read_pickle(
    "data/ariel_pqdm/wave_to_apriori_scale.pkl"
    )
    sample_submission = pd.read_csv(
        "data/sample_submission.csv",
        index_col="planet_id",
    )

    predictions_spectra = np.repeat(np.array(predictions_spectra), 283).reshape(
        (len(predictions_spectra), 283)
    )
    predictions_spectra = predictions_spectra.clip(0)

    sigmas = np.ones_like(predictions_spectra) * 0.000140985

    submission = pd.DataFrame(
        np.concatenate([predictions_spectra, sigmas], axis=1),
        columns=sample_submission.columns,
    )
    submission.index = sample_submission.index

    for wave, scale in wave_to_apriori_scale.items():
        if 0.99 < scale < 1.01:
            scale = 1.0
        if wave in ['wl_2']:
            scale = 0.99
        if wave in ['wl_133', 'wl_134']:
            scale = 1.01
        scale = np.clip(scale, 0.993, 1.007)
        submission[wave] *= scale

    submission.to_csv("submission.csv")
    print(submission)