import enum
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.paths import get_data_dir

# Date of the 7.8-magnitude earthquake that struck coastal Ecuador.
_ECUADOR_EARTHQUAKE_DATE = pd.Timestamp("2016-04-16")


class EarthquakeEncoding(enum.StrEnum):
    """How to encode elapsed time since the 2016 Ecuador earthquake.

    DECAY:  exp(-days_since / tau), equals 1.0 on the event day and decays toward 0.
            Requires an earthquake_tau parameter (days). Good default when you have
            a prior on the recovery timescale.
    LINEAR: days_since / 365, equals 0 on the event day and grows linearly.
            No hyperparameter; lets the model learn the decay shape via its FFN layers.
    """

    DECAY = enum.auto()
    LINEAR = enum.auto()


def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(pd.to_datetime(df["date"]))


STORE_FEATURE_COLS = ("city", "state", "type", "cluster")

HOLIDAY_FEATURE_COLS = (
    "national_holiday",
    "event",
    "bridge",
    "work_day",
    "additional",
    "regional_holiday",
    "local_holiday",
)


class StoreData(Dataset):
    """Store sales dataset for the Kaggle Store Sales forecasting competition.

    Loads all CSVs on construction and builds a [T, n_stores, n_families] sales
    tensor plus auxiliary feature tensors. Acts as a sliding-window Dataset:
    each item is an (input, target) pair. By default all feature groups are
    enabled.

    Input shape per item:
        [window_lags, n_stores, n_families, n_input_channels]  (flatten_output=False)
        [window_lags, n_stores, n_families * n_input_channels] (flatten_output=True)
    Feature axis order: sales (1), promotion (1), date (n_date_features),
    oil (1), store features (n_store_features), holiday (n_holiday_features).
    Sales and promotion vary per family; all other features are broadcast.
    Target shape is always [output_lags, n_stores, n_families] (sales only).

    Attributes:
        train, test, sample_submission, stores, oil, holidays: raw DataFrames.
        sales_tensor: float32 Tensor of shape [T, 54, 33].
        date_features_tensor: float32 Tensor of shape [T, n_date_features].
            When date_features=True (5 cols): days since first training date,
            sin/cos day-of-week (period 7), sin/cos day-of-year (period 365.25).
            When payday_features=True (2 cols): is_15th, is_month_end.
            When earthquake_encoding is set (1 col): earthquake proximity signal.
            Columns appear in the order listed above.
        n_date_features: total number of date feature columns (0–8).
        families: Index mapping column position -> product family name.
        oil_tensor: float32 Tensor of shape [T] when include_oil=True, else None.
            NaN gaps (weekends/holidays) are filled via forward-fill then back-fill.
        include_oil: whether oil price is included in the input.
        promotion_tensor: float32 Tensor of shape [T, 54, 33], or None if
            include_onpromotion is False.
        store_feature_tensor: float32 Tensor of shape [54, n_store_features],
            or None if no store features were requested.
        store_feature_encoders: dict mapping column name -> sorted unique values
            used for label encoding (categorical columns only).
        holiday_tensor: float32 Tensor of shape [T, n_stores, n_holiday_features],
            or None if no holiday features were requested. National-scope features
            (national_holiday, event, bridge, work_day, additional) are broadcast
            across all stores. Per-store features (regional_holiday, local_holiday)
            are matched by state and city respectively.
        n_holiday_features: number of holiday feature channels (0 when disabled).
        flatten_output: if True, __getitem__ returns a 3D input tensor with the
            family and feature axes merged into one.
    """

    def __init__(
        self,
        window_lags: int = 60,
        output_lags: int = 16,
        data_dir: Path | None = None,
        copy: bool = False,
        dtype: torch.dtype = torch.float32,
        date_features: bool = True,
        payday_features: bool = True,
        earthquake_encoding: EarthquakeEncoding | None = EarthquakeEncoding.DECAY,
        earthquake_tau: float = 30.0,
        include_oil: bool = True,
        include_onpromotion: bool = True,
        store_feature_cols: Iterable[str] | None = STORE_FEATURE_COLS,
        holiday_features: Iterable[str] | None = HOLIDAY_FEATURE_COLS,
        flatten_output: bool = False,
    ) -> None:
        """Load data and build the sales tensor.

        Args:
            window_lags: length of the input window fed to the model.
            output_lags: length of the prediction horizon (competition uses 16).
            data_dir: directory containing the competition CSVs. Defaults to
                <repo_root>/data/store-sales-time-series-forecasting.
            copy: if True, copy the underlying numpy array so the tensor is
                writable; costs ~2× memory.
            date_features: if True, append days-since-start, sin/cos day-of-week,
                and sin/cos day-of-year (5 columns) to each input timestep.
            payday_features: if True, append is_15th and is_month_end binary
                indicators (2 columns) to each input timestep.
            earthquake_encoding: how to encode time since the 2016 Ecuador earthquake
                (1 column). DECAY uses exp(-days_since / earthquake_tau); LINEAR uses
                days_since / 365. Set to None to omit the feature entirely.
            earthquake_tau: decay time constant in days for EarthquakeEncoding.DECAY.
                Ignored when earthquake_encoding is LINEAR or None.
            include_oil: if True, oil price is appended as an extra feature channel.
            include_onpromotion: if True, per-family promotion flags are appended as
                an extra feature channel. The target window is always sales-only.
            store_feature_cols: columns from ("city", "state", "type", "cluster") to
                include as static per-store feature channels. Defaults to all four.
                Categoricals are one-hot encoded; "cluster" is a single numeric column.
            holiday_features: subset of HOLIDAY_FEATURE_COLS to include as binary
                feature channels. Defaults to all seven. National-scope features
                (national_holiday, event, bridge, work_day, additional) are broadcast
                across all 54 stores; per-store features (regional_holiday,
                local_holiday) are matched by store state and city respectively.
                Transferred holidays are treated as ordinary days.
            flatten_output: if True, __getitem__ returns a 3D input tensor of shape
                [window_lags, n_stores, n_families * n_input_channels] by merging the
                family and feature axes. Useful for models that expect a flat last
                dimension. Default False returns the full 4D shape.
        """
        if data_dir is None:
            data_dir = get_data_dir() / "store-sales-time-series-forecasting"
        self.dtype = dtype
        self.include_oil = include_oil
        self.flatten_output = flatten_output
        self.train = self._load_train(data_dir)
        self.test = self._load_test(data_dir)
        self.sample_submission = self._load_sample_submission(data_dir)
        self.stores = self._load_stores(data_dir)
        self.oil = self._load_oil(data_dir)
        self.holidays = self._load_holidays(data_dir)

        self.window_lags = window_lags
        self.output_lags = output_lags
        self.sales_tensor, self.families = self._setup_tensor(
            self.train, self.stores, dtype, copy
        )
        dates = pd.DatetimeIndex(self.train.index.unique().sort_values())
        self.date_features_tensor = self._setup_date_features(
            dates,
            date_features,
            payday_features,
            earthquake_encoding,
            earthquake_tau,
            dtype,
        )
        self.n_date_features: int = self.date_features_tensor.shape[1]
        self.oil_tensor: Tensor | None = (
            self._setup_oil_tensor(self.oil, self.train, dtype) if include_oil else None
        )
        self.promotion_tensor: Tensor | None = (
            self._setup_promotion_tensor(self.train, self.stores, dtype, copy)
            if include_onpromotion
            else None
        )
        self._len = self.sales_tensor.shape[0] - window_lags - output_lags

        requested_cols = list(store_feature_cols) if store_feature_cols else []
        invalid = set(requested_cols) - set(STORE_FEATURE_COLS)
        if invalid:
            raise ValueError(
                f"Unknown store feature column(s): {invalid}. "
                f"Valid options: {STORE_FEATURE_COLS}"
            )
        self.store_feature_tensor, self.store_feature_encoders = (
            self._build_store_features(self.stores, requested_cols, dtype)
        )

        requested_holiday_features = list(holiday_features) if holiday_features else []
        invalid_holiday = set(requested_holiday_features) - set(HOLIDAY_FEATURE_COLS)
        if invalid_holiday:
            raise ValueError(
                f"Unknown holiday feature(s): {invalid_holiday}. "
                f"Valid options: {HOLIDAY_FEATURE_COLS}"
            )
        self.holiday_tensor: Tensor | None = self._setup_holiday_tensor(
            self.holidays,
            self.train,
            self.stores,
            requested_holiday_features,
            dtype,
        )

    @staticmethod
    def _build_store_features(
        stores: pd.DataFrame,
        cols: list[str],
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, dict[str, list]]:
        """Build a [n_stores, n_features] tensor of encoded store metadata.

        Stores are taken in store_nbr order (ascending index). Categorical
        columns (city, state, type) are one-hot encoded using sorted unique
        values. The numeric column (cluster) is a single column cast directly.

        Returns:
            (tensor, encoders) where encoders maps each categorical column name
            to its sorted list of unique values (index = one-hot position).
            tensor is None when cols is empty.
        """
        if not cols:
            return None, {}

        categorical_cols = {"city", "state", "type"}
        sorted_stores = stores.sort_index()
        encoders: dict[str, list] = {}
        blocks: list[np.ndarray] = []
        for col in cols:
            series = sorted_stores[col]
            if col in categorical_cols:
                unique_values = sorted(series.unique().tolist())
                encoders[col] = unique_values
                label_map = {v: i for i, v in enumerate(unique_values)}
                indices = np.array([label_map[v] for v in series], dtype=np.intp)
                blocks.append(np.eye(len(unique_values), dtype="float32")[indices])
            else:
                blocks.append(series.to_numpy(dtype="float32")[:, None])

        arr = np.concatenate(blocks, axis=1)
        return torch.from_numpy(arr).to(dtype), encoders  # type: ignore[attr-defined]

    @staticmethod
    def _setup_holiday_tensor(
        holidays: pd.DataFrame,
        train: pd.DataFrame,
        stores: pd.DataFrame,
        requested_features: list[str],
        dtype: torch.dtype = torch.float32,
    ) -> Tensor | None:
        """Build a [T, n_stores, n_holiday_features] tensor of binary holiday indicators.

        National-scope features are broadcast across all stores. Per-store features
        (regional_holiday, local_holiday) match the store's state or city against
        the holiday's locale_name. transferred=True rows are treated as ordinary days;
        type=Transfer rows capture the actual rescheduled celebration date.

        Args:
            holidays: DataFrame indexed by date with columns type, locale,
                locale_name, transferred.
            train: training DataFrame indexed by date; used to derive T and date order.
            stores: stores DataFrame indexed by store_nbr with city, state columns.
            requested_features: ordered subset of HOLIDAY_FEATURE_COLS.

        Returns:
            Float32 Tensor of shape [T, n_stores, len(requested_features)], or None
            when requested_features is empty.
        """
        if not requested_features:
            return None

        train_dates = train.index.unique().sort_values()
        n_times = len(train_dates)
        n_stores = stores.shape[0]
        sorted_stores = stores.sort_index()

        def _active_national_dates(holiday_type: str) -> set:
            """Dates where a national-scope type is active (not a transferred-away day)."""
            return set(holidays.index[holidays["type"] == holiday_type])

        def _broadcast(date_set: set) -> np.ndarray:
            """[T, n_stores] array with all stores sharing the same value."""
            active_mask = train_dates.isin(date_set).astype("float32")
            return np.tile(active_mask[:, None], (1, n_stores))

        def _per_store_locale(locale: str, store_col: str) -> np.ndarray:
            """[T, n_stores] array with per-store holiday indicators matched by locale."""
            not_transferred = (
                (holidays["type"] == "Holiday")
                & (holidays["locale"] == locale)
                & ~holidays["transferred"]
            )
            transfer_rows = (holidays["type"] == "Transfer") & (
                holidays["locale"] == locale
            )
            locale_names = holidays.loc[not_transferred | transfer_rows, "locale_name"]
            date_to_locale_names = (
                locale_names.groupby(level=0).apply(frozenset).to_dict()
            )
            store_locale_values = sorted_stores[store_col].tolist()
            arr = np.zeros((n_times, n_stores), dtype="float32")
            for time_idx, date in enumerate(train_dates):
                if date in date_to_locale_names:
                    active_locales = date_to_locale_names[date]
                    for store_idx, store_locale in enumerate(store_locale_values):
                        if store_locale in active_locales:
                            arr[time_idx, store_idx] = 1.0
            return arr

        # national_holiday: type=Holiday national (not transferred away) + type=Transfer national
        def _national_holiday_dates() -> set:
            not_transferred = (
                (holidays["type"] == "Holiday")
                & (holidays["locale"] == "National")
                & ~holidays["transferred"]
            )
            transfer_rows = (holidays["type"] == "Transfer") & (
                holidays["locale"] == "National"
            )
            return set(holidays.index[not_transferred | transfer_rows])

        columns: list[np.ndarray] = []
        for feature in requested_features:
            if feature == "national_holiday":
                columns.append(_broadcast(_national_holiday_dates()))
            elif feature == "event":
                columns.append(_broadcast(_active_national_dates("Event")))
            elif feature == "bridge":
                columns.append(_broadcast(_active_national_dates("Bridge")))
            elif feature == "work_day":
                columns.append(_broadcast(_active_national_dates("Work Day")))
            elif feature == "additional":
                columns.append(_broadcast(_active_national_dates("Additional")))
            elif feature == "regional_holiday":
                columns.append(_per_store_locale("Regional", "state"))
            elif feature == "local_holiday":
                columns.append(_per_store_locale("Local", "city"))

        arr_3d = np.stack(columns, axis=-1)  # [T, n_stores, n_features]
        return torch.from_numpy(arr_3d).to(dtype)  # type: ignore[attr-defined]

    @staticmethod
    def _load_train(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "train.csv"))

    @staticmethod
    def _load_test(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "test.csv"))

    @staticmethod
    def _load_sample_submission(data_dir: Path) -> pd.DataFrame:
        return pd.read_csv(data_dir / "sample_submission.csv")

    @staticmethod
    def _load_stores(data_dir: Path) -> pd.DataFrame:
        return pd.read_csv(data_dir / "stores.csv").set_index("store_nbr")

    @staticmethod
    def _load_oil(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "oil.csv"))

    @staticmethod
    def _load_holidays(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "holidays_events.csv"))

    @staticmethod
    def _setup_tensor(
        train: pd.DataFrame,
        stores: pd.DataFrame,
        dtype: torch.dtype = torch.float32,
        copy: bool = False,
    ) -> tuple[Tensor, pd.Index]:
        num_stores = stores.shape[0]
        pivot = train.pivot(columns=("store_nbr", "family"), values="sales").sort_index(
            axis="columns"
        )
        families: pd.Index = pivot.columns.get_level_values("family").unique()
        num_families = len(families)
        arr = pivot.to_numpy().reshape(pivot.shape[0], num_stores, num_families)
        if copy:
            arr = arr.copy()
        return torch.from_numpy(arr).to(dtype), families  # type: ignore[attr-defined]

    @staticmethod
    def _setup_date_features(
        dates: pd.DatetimeIndex,
        date_features: bool = True,
        payday_features: bool = True,
        earthquake_encoding: EarthquakeEncoding | None = EarthquakeEncoding.DECAY,
        earthquake_tau: float = 30.0,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Build a [T, n_date_features] tensor of enabled date features.

        date_features group (5 cols): days since first date, sin/cos day-of-week,
            sin/cos day-of-year.
        payday_features group (2 cols): is_15th, is_month_end.
        earthquake group (1 col): time-since-earthquake signal; see EarthquakeEncoding.
        Returns a [T, 0] empty tensor when all groups are disabled.
        """
        parts: list[Tensor] = []
        date_series = dates.to_series()
        if date_features:
            epoch = dates[0]
            days = torch.tensor(
                [(d - epoch).days for d in dates], dtype=dtype
            ).unsqueeze(1)
            two_pi = 2.0 * math.pi
            dow = torch.tensor(date_series.dt.dayofweek.to_numpy(), dtype=dtype)
            doy = torch.tensor(date_series.dt.dayofyear.to_numpy(), dtype=dtype)
            parts += [
                days,
                (two_pi * dow / 7).sin().unsqueeze(1),
                (two_pi * dow / 7).cos().unsqueeze(1),
                (two_pi * doy / 365.25).sin().unsqueeze(1),
                (two_pi * doy / 365.25).cos().unsqueeze(1),
            ]
        if payday_features:
            parts += [
                torch.tensor(
                    (date_series.dt.day == 15).to_numpy(), dtype=dtype
                ).unsqueeze(1),
                torch.tensor(
                    date_series.dt.is_month_end.to_numpy(), dtype=dtype
                ).unsqueeze(1),
            ]
        if earthquake_encoding is not None:
            days_since = torch.tensor(
                [(d - _ECUADOR_EARTHQUAKE_DATE).days for d in dates], dtype=dtype
            )
            if earthquake_encoding == EarthquakeEncoding.DECAY:
                feature = torch.where(
                    days_since >= 0,
                    torch.exp(-days_since.clamp(min=0) / earthquake_tau),
                    torch.zeros_like(days_since),
                )
            else:  # LINEAR
                feature = days_since.clamp(min=0) / 365.0
            parts.append(feature.unsqueeze(1))
        if not parts:
            return torch.zeros(len(dates), 0, dtype=dtype)
        return torch.cat(parts, dim=1)

    @staticmethod
    def _setup_oil_tensor(
        oil: pd.DataFrame,
        train: pd.DataFrame,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Align oil prices to train dates and fill NaN gaps.

        Oil is reported only on trading days; weekends and holidays are NaN.
        Forward-fill propagates the last known price, then back-fill covers any
        leading NaNs at the start of the series.
        """
        train_dates = train.index.unique().sort_values()
        aligned = oil["dcoilwtico"].reindex(train_dates).ffill().bfill()
        arr = aligned.to_numpy(dtype="float32")
        return torch.from_numpy(arr).to(dtype)  # type: ignore[attr-defined]

    @staticmethod
    def _setup_promotion_tensor(
        train: pd.DataFrame,
        stores: pd.DataFrame,
        dtype: torch.dtype = torch.float32,
        copy: bool = False,
    ) -> Tensor:
        """Build a [T, n_stores, n_families] tensor from the onpromotion column.

        Column order matches sales_tensor — both pivot on (store_nbr, family)
        sorted the same way, so indices align.
        """
        num_stores = stores.shape[0]
        pivot = train.pivot(
            columns=("store_nbr", "family"), values="onpromotion"
        ).sort_index(axis="columns")
        num_families = pivot.columns.get_level_values("family").nunique()
        arr = pivot.to_numpy().reshape(pivot.shape[0], num_stores, num_families)
        if copy:
            arr = arr.copy()
        return torch.from_numpy(arr).to(dtype)  # type: ignore[attr-defined]

    def _build_window_parts(self, start: int, mid: int) -> list[Tensor]:
        """Build feature parts for the window [start:mid], each [T, S, F, k].

        Feature order: sales, promotion, date, oil, store, holiday.
        Per-family features (sales, promo) naturally vary across F.
        All other features are broadcast across the family axis via expand.
        """
        _, n_stores, n_families = self.sales_tensor.shape
        T = mid - start
        parts: list[Tensor] = [self.sales_tensor[start:mid].unsqueeze(-1)]

        if self.promotion_tensor is not None:
            parts.append(self.promotion_tensor[start:mid].unsqueeze(-1))

        if self.n_date_features > 0:
            parts.append(
                self.date_features_tensor[start:mid]
                .view(T, 1, 1, self.n_date_features)
                .expand(T, n_stores, n_families, -1)
            )

        if self.oil_tensor is not None:
            parts.append(
                self.oil_tensor[start:mid]
                .view(T, 1, 1, 1)
                .expand(T, n_stores, n_families, 1)
            )

        if self.store_feature_tensor is not None:
            parts.append(
                self.store_feature_tensor.view(1, n_stores, 1, -1).expand(
                    T, n_stores, n_families, -1
                )
            )

        if self.holiday_tensor is not None:
            parts.append(
                self.holiday_tensor[start:mid]
                .unsqueeze(2)
                .expand(T, n_stores, n_families, -1)
            )

        return parts

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return (input, target) windows.

        input shape (flatten_output=False): [window_lags, n_stores, n_families, n_input_channels]
        input shape (flatten_output=True):  [window_lags, n_stores, n_families * n_input_channels]
        target shape: [output_lags, n_stores, n_families] (sales only).
        """
        start = index
        mid = index + self.window_lags
        end = mid + self.output_lags

        x = torch.cat(self._build_window_parts(start, mid), dim=-1)  # type: ignore[attr-defined]
        if self.flatten_output:
            x = x.flatten(-2)

        return x, self.sales_tensor[mid:end]

    def __len__(self) -> int:
        """Number of stride-1 sliding windows; excludes the final output_lags days."""
        return self._len

    @property
    def n_store_features(self) -> int:
        """Number of extra store feature channels appended to each input timestep."""
        if self.store_feature_tensor is None:
            return 0
        return self.store_feature_tensor.shape[1]

    @property
    def n_holiday_features(self) -> int:
        """Number of holiday feature channels appended to each input timestep."""
        if self.holiday_tensor is None:
            return 0
        return self.holiday_tensor.shape[2]

    @property
    def n_input_channels(self) -> int:
        """Size of the feature axis in __getitem__ input: [T, n_stores, n_families, n_input_channels]."""
        return (
            1  # sales
            + (1 if self.promotion_tensor is not None else 0)
            + self.n_date_features
            + (1 if self.include_oil else 0)
            + self.n_store_features
            + self.n_holiday_features
        )

    def __repr__(self) -> str:
        T, n_stores, n_families = self.sales_tensor.shape
        if self.flatten_output:
            input_shape = f"[window, {n_stores}, {n_families * self.n_input_channels}]"
        else:
            input_shape = f"[window, {n_stores}, {n_families}, {self.n_input_channels}]"
        return (
            f"StoreData(T={T}, n_stores={n_stores}, n_families={n_families}, "
            f"input_shape={input_shape}, "
            f"n_date_features={self.n_date_features}, "
            f"include_oil={self.include_oil}, n_store_features={self.n_store_features}, "
            f"n_holiday_features={self.n_holiday_features}, "
            f"flatten_output={self.flatten_output}, "
            f"window_lags={self.window_lags}, output_lags={self.output_lags})"
        )
