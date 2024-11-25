import numpy as np
import pandas as pd
import itertools
from pqdm.processes import pqdm
from astropy.stats import sigma_clip


class Calibrator:
    cut_inf = 39
    cut_sup = 321
    sensor_to_sizes_dict = {
        "AIRS-CH0": [[11250, 32, 356], [1, 32, cut_sup - cut_inf]],
        "FGS1": [[135000, 32, 32], [1, 32, 32]],
    }
    sensor_to_linear_corr_dict = {"AIRS-CH0": (6, 32, 356), "FGS1": (6, 32, 32)}

    def __init__(self, dataset, planet_id, sensor):
        self.dataset = dataset
        self.planet_id = str(planet_id)
        self.sensor = sensor

    def _apply_linear_corr(self, linear_corr, clean_signal):
        linear_corr = np.flip(linear_corr, axis=0)
        for x, y in itertools.product(
            range(clean_signal.shape[1]), range(clean_signal.shape[2])
        ):
            poli = np.poly1d(linear_corr[:, x, y])
            clean_signal[:, x, y] = poli(clean_signal[:, x, y])
        return clean_signal

    def _clean_dark(self, signal, dark, dt):
        dark = np.tile(dark, (signal.shape[0], 1, 1))
        signal -= dark * dt[:, np.newaxis, np.newaxis]
        return signal

    def get_calibrated_signal(self):
        adc_info = pd.read_csv(f"data/{self.dataset}_adc_info.csv", index_col="planet_id")
        adc_info.index = adc_info.index.astype(str)

        signal = pd.read_parquet(
            f"data/{self.dataset}/{self.planet_id}/{self.sensor}_signal.parquet"
        ).to_numpy()
        dark_frame = pd.read_parquet(
            f"data/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/dark.parquet",
            engine="pyarrow",
        ).to_numpy()
        dead_frame = pd.read_parquet(
            f"data/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/dead.parquet",
            engine="pyarrow",
        ).to_numpy()
        flat_frame = pd.read_parquet(
            f"data/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/flat.parquet",
            engine="pyarrow",
        ).to_numpy()
        linear_corr = (
            pd.read_parquet(
                f"data/{self.dataset}/{self.planet_id}/{self.sensor}_calibration/linear_corr.parquet"
            )
            .values.astype(np.float64)
            .reshape(self.sensor_to_linear_corr_dict[self.sensor])
        )

        signal = signal.reshape(self.sensor_to_sizes_dict[self.sensor][0])
        gain = adc_info.loc[self.planet_id, f"{self.sensor}_adc_gain"] 
        offset = adc_info.loc[self.planet_id, f"{self.sensor}_adc_offset"] 
        signal = signal / gain + offset

        hot = sigma_clip(dark_frame, sigma=5, maxiters=5).mask

        if self.sensor == "AIRS-CH0":
            signal = signal[:, :, self.cut_inf : self.cut_sup]
            dt = np.ones(len(signal)) * 0.1
            dt[1::2] += 4.5  
            linear_corr = linear_corr[:, :, self.cut_inf : self.cut_sup]
            dark_frame = dark_frame[:, self.cut_inf : self.cut_sup]
            dead_frame = dead_frame[:, self.cut_inf : self.cut_sup]
            flat_frame = flat_frame[:, self.cut_inf : self.cut_sup]
            hot = hot[:, self.cut_inf : self.cut_sup]
        elif self.sensor == "FGS1":
            dt = np.ones(len(signal)) * 0.1
            dt[1::2] += 0.1

        signal = signal.clip(0)  
        linear_corr_signal = self._apply_linear_corr(linear_corr, signal)
        signal = self._clean_dark(linear_corr_signal, dark_frame, dt)

        flat = flat_frame.reshape(self.sensor_to_sizes_dict[self.sensor][1])
        flat[dead_frame.reshape(self.sensor_to_sizes_dict[self.sensor][1])] = np.nan
        flat[hot.reshape(self.sensor_to_sizes_dict[self.sensor][1])] = np.nan
        signal = signal / flat
        return signal


class Preprocessor:
    sensor_to_binning = {"AIRS-CH0": 30, "FGS1": 30 * 12}
    sensor_to_binned_dict = {
        "AIRS-CH0": [11250 // sensor_to_binning["AIRS-CH0"] // 2, 282],
        "FGS1": [135000 // sensor_to_binning["FGS1"] // 2],
    }

    def __init__(self, dataset, planet_id, sensor):
        self.dataset = dataset
        self.planet_id = planet_id
        self.sensor = sensor
        self.binning = self.sensor_to_binning[sensor]

    def preprocess_signal(self):
        signal = Calibrator(
            dataset=self.dataset, planet_id=self.planet_id, sensor=self.sensor
        ).get_calibrated_signal()

        if self.sensor == "AIRS-CH0":
            signal = signal[:, 10:22, :]
        elif self.sensor == "FGS1":
            signal = signal[:, 10:22, 10:22]
            signal = signal.reshape(
                signal.shape[0], signal.shape[1] * signal.shape[2]
            )

        mean_signal = np.nanmean(signal, axis=1)
        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        binned = np.zeros((self.sensor_to_binned_dict[self.sensor]))
        for j in range(cds_signal.shape[0] // self.binning):
            binned[j] = cds_signal[
                j * self.binning : j * self.binning + self.binning
            ].mean(axis=0)

        if self.sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        return binned


def preprocessor(x):
    return Preprocessor(**x).preprocess_signal()

def preprocess_signals(dataset):
    adc_info = pd.read_csv(f"data/{dataset}_adc_info.csv", index_col="planet_id")
    planet_ids = adc_info.index

    args_fgs1 = [
        dict(dataset=dataset, planet_id=planet_id, sensor="FGS1")
        for planet_id in planet_ids
    ]
    preprocessed_signal_fgs1 = pqdm(args_fgs1, preprocessor, n_jobs=4)

    args_airs_ch0 = [
        dict(dataset=dataset, planet_id=planet_id, sensor="AIRS-CH0")
        for planet_id in planet_ids
    ]
    preprocessed_signal_airs_ch0 = pqdm(args_airs_ch0, preprocessor, n_jobs=4)

    preprocessed_signal = np.concatenate(
        [np.stack(preprocessed_signal_fgs1), np.stack(preprocessed_signal_airs_ch0)], axis=2
    )
    return preprocessed_signal
