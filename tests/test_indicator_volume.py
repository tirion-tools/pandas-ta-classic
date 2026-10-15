from unittest import TestCase

import numpy as np
from pandas import DataFrame, Series, date_range

import pandas_ta_classic as pandas_ta
from tests.assertions import IndicatorSpec, assert_indicator_standard, assert_talib
from tests.config import get_sample_data

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


_VCLIMAX = "VCLIMAX_50_20_2.0_3_8.0_5_0.6"
_VCLIMAX_FT = "VCLIMAXft_50_20_2.0_3_8.0_5_0.6"


def _climax_frame(bars: list[tuple[float, float, float, float]]) -> dict[str, Series]:
    """55 flat bars (high 101, low 99, close 100, volume 1000), then *bars* as (high, low, close, volume)."""
    rows = [(101.0, 99.0, 100.0, 1000.0)] * 55 + bars
    index = date_range("2020-01-01", periods=len(rows), freq="D")
    high, low, close, volume = (Series([row[i] for row in rows], index=index) for i in range(4))
    return {"high": high, "low": low, "close": close, "volume": volume}


# Three down closes, then a bar on three times the volume that makes a new low and
# closes at 90 % of its range, then a bar with a higher low and a higher close.
_SELLING = [
    (100.0, 98.0, 99.0, 1000.0),
    (99.0, 97.0, 98.0, 1000.0),
    (98.0, 96.0, 97.0, 1000.0),
    (98.0, 94.0, 97.6, 3000.0),
    (99.0, 95.0, 98.0, 1500.0),
]
# The mirror: three up closes, a heavy new high closing at 10 % of its range, a lower high and close.
_BUYING = [
    (102.0, 100.0, 101.0, 1000.0),
    (103.0, 101.0, 102.0, 1000.0),
    (104.0, 102.0, 103.0, 1000.0),
    (106.0, 102.0, 102.4, 3000.0),
    (105.0, 101.0, 102.0, 1500.0),
]


class TestVolume(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume_ = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume_
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ad(self):
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.AD(self.high, self.low, self.close, self.volume_),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ad,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="AD",
                none_arg_idx=None,
            ),
        )

    def test_ad_open(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ad,
                args=[self.high, self.low, self.close, self.volume_, self.open],
                expected_name="ADo",
                none_arg_idx=None,
            ),
        )

    def test_adosc(self):
        result = pandas_ta.adosc(self.high, self.low, self.close, self.volume_, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.ADOSC(self.high, self.low, self.close, self.volume_),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.adosc,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="ADOSC_3_10",
            ),
        )

    def test_aobv(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.aobv,
                args=[self.close, self.volume_],
                expected_name="AOBVe_4_12_2_2_2",
                expected_type=DataFrame,
                expected_columns=[
                    "OBV",
                    "OBV_min_2",
                    "OBV_max_2",
                    "OBVe_4",
                    "OBVe_12",
                    "AOBV_LR_2",
                    "AOBV_SR_2",
                ],
                none_arg_idx=None,
            ),
        )

    def test_cmf(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cmf,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="CMF_20",
            ),
        )

    def test_efi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.efi,
                args=[self.close, self.volume_],
                expected_name="EFI_13",
            ),
        )

    def test_emv(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.emv,
                args=[self.high, self.low, self.volume_],
                expected_name="EMV",
            ),
        )

    def test_eom(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.eom,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="EOM_14_100000000",
            ),
        )

    def test_kvo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.kvo,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="KVO_34_55_13",
                expected_type=DataFrame,
                expected_columns=["KVO_34_55_13", "KVOs_34_55_13"],
            ),
        )

    def test_marketfi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.marketfi,
                args=[self.high, self.low, self.volume_],
                expected_name="MARKETFI",
            ),
        )
        self.assertIsNone(pandas_ta.marketfi(self.high, None, self.volume_))
        self.assertIsNone(pandas_ta.marketfi(self.high, self.low, None))

    def test_mfi(self):
        result = pandas_ta.mfi(self.high, self.low, self.close, self.volume_, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.MFI(self.high, self.low, self.close, self.volume_),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mfi,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="MFI_14",
            ),
        )

    def test_nvi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.nvi,
                args=[self.close, self.volume_],
                expected_name="NVI_1",
            ),
        )

    def test_obv(self):
        result = pandas_ta.obv(self.close, self.volume_, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.OBV(self.close, self.volume_),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.obv,
                args=[self.close, self.volume_],
                expected_name="OBV",
                none_arg_idx=None,
            ),
        )

    def test_pvi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvi,
                args=[self.close, self.volume_],
                expected_name="PVI_1",
            ),
        )

    def test_pvol(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvol,
                args=[self.close, self.volume_],
                expected_name="PVOL",
                none_arg_idx=None,
            ),
        )

    def test_pvr(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvr,
                args=[self.close, self.volume_],
                expected_name="PVR",
                none_arg_idx=None,
            ),
        )
        # sample indicator values from SPY
        # Bar 0 has no difference to rank; it read 1, a rising close, before.
        self.assertTrue(result.isna().iloc[0])
        self.assertEqual(result.iloc[1], 3)
        self.assertEqual(result.iloc[4], 2)
        self.assertEqual(result.iloc[6], 4)

    def test_pvt(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvt,
                args=[self.close, self.volume_],
                expected_name="PVT",
                none_arg_idx=None,
            ),
        )

    def test_vp(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vp,
                args=[self.close, self.volume_],
                expected_name="VP_10",
                expected_type=DataFrame,
                expected_columns=[
                    "low_close",
                    "mean_close",
                    "high_close",
                    "pos_volume",
                    "neg_volume",
                    "total_volume",
                ],
                none_arg_idx=None,
            ),
        )

    def test_vp_sort_close(self):
        chronological = pandas_ta.vp(self.close, self.volume_)
        by_price = pandas_ta.vp(self.close, self.volume_, sort_close=True)
        self.assertIsInstance(by_price, DataFrame)
        self.assertEqual(list(by_price.columns), list(chronological.columns))
        self.assertEqual(by_price.name, "VP_10")

        # sort_close bins by price, so the bins are ordered and disjoint and the
        # profile differs from the chronological split.
        self.assertTrue(by_price["mean_close"].is_monotonic_increasing)
        self.assertTrue((by_price["low_close"] <= by_price["mean_close"]).all())
        self.assertTrue((by_price["mean_close"] <= by_price["high_close"]).all())
        self.assertTrue((by_price["high_close"].iloc[:-1].to_numpy() <= by_price["low_close"].iloc[1:].to_numpy()).all())
        self.assertFalse(by_price["total_volume"].equals(chronological["total_volume"]))

        # Both splits account for the same total volume.
        self.assertAlmostEqual(
            float(by_price["total_volume"].sum()),
            float(chronological["total_volume"].sum()),
            places=6,
        )

        # width controls the number of bins on both paths.
        self.assertLessEqual(len(pandas_ta.vp(self.close, self.volume_, width=5, sort_close=True)), 5)

    def test_vfi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vfi,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="VFI_130",
            ),
        )

    def test_vosc(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vosc,
                args=[self.volume_],
                expected_name="VOSC_14_28",
            ),
        )

    def test_wad(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.wad,
                args=[self.high, self.low, self.close],
                expected_name="WAD",
            ),
        )
        self.assertIsNone(pandas_ta.wad(self.high, None, self.close))
        self.assertIsNone(pandas_ta.wad(self.high, self.low, None))

    def test_vclimax(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vclimax,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name=_VCLIMAX,
                expected_type=DataFrame,
                expected_columns=["VCLIMAXr_50", "VCLIMAXp", _VCLIMAX, _VCLIMAX_FT],
            ),
        )

    def test_vclimax_selling_climax_and_follow_through(self):
        result = pandas_ta.vclimax(**_climax_frame(_SELLING))
        climax = result.iloc[-2]
        self.assertAlmostEqual(climax["VCLIMAXr_50"], 3.0)
        self.assertAlmostEqual(climax["VCLIMAXp"], 0.9)
        self.assertEqual(climax[_VCLIMAX], 1.0)
        self.assertEqual(result[_VCLIMAX].iloc[50:-2].abs().sum(), 0.0)
        self.assertEqual(result[_VCLIMAX].iloc[-1], 0.0)
        self.assertEqual(result[_VCLIMAX_FT].iloc[-1], 1.0)
        self.assertEqual(result[_VCLIMAX_FT].iloc[51:-1].abs().sum(), 0.0)

    def test_vclimax_buying_climax_and_follow_through(self):
        result = pandas_ta.vclimax(**_climax_frame(_BUYING))
        self.assertAlmostEqual(result["VCLIMAXp"].iloc[-2], 0.1)
        self.assertEqual(result[_VCLIMAX].iloc[-2], -1.0)
        self.assertEqual(result[_VCLIMAX_FT].iloc[-1], -1.0)

    def test_vclimax_no_follow_through_without_a_higher_low(self):
        bars = _SELLING[:-1] + [(99.0, 93.0, 98.0, 1500.0)]
        result = pandas_ta.vclimax(**_climax_frame(bars))
        self.assertEqual(result[_VCLIMAX].iloc[-2], 1.0)
        self.assertEqual(result[_VCLIMAX_FT].iloc[-1], 0.0)

    def test_vclimax_heavy_bar_closing_on_its_low_is_no_climax(self):
        bars = _SELLING[:3] + [(98.0, 94.0, 94.4, 3000.0)]
        result = pandas_ta.vclimax(**_climax_frame(bars))
        self.assertAlmostEqual(result["VCLIMAXp"].iloc[-1], 0.1)
        self.assertEqual(result[_VCLIMAX].iloc[-1], 0.0)

    def test_vclimax_needs_the_volume(self):
        bars = _SELLING[:3] + [(98.0, 94.0, 97.6, 1900.0)]
        self.assertEqual(pandas_ta.vclimax(**_climax_frame(bars))[_VCLIMAX].iloc[-1], 0.0)
        self.assertEqual(pandas_ta.vclimax(**_climax_frame(bars), mult=1.9)["VCLIMAX_50_20_1.9_3_8.0_5_0.6"].iloc[-1], 1.0)

    def test_vclimax_drop_stands_in_for_the_run(self):
        # Two down closes only, but 100 -> 91.5 (-8.5 %) over the five bars before the climax.
        bars = [
            (100.0, 93.0, 94.0, 1000.0),
            (95.0, 92.0, 93.0, 1000.0),
            (94.0, 92.5, 93.5, 1000.0),
            (94.0, 91.5, 92.5, 1000.0),
            (93.0, 90.5, 91.5, 1000.0),
            (93.0, 89.0, 92.6, 3000.0),
        ]
        result = pandas_ta.vclimax(**_climax_frame(bars))
        self.assertEqual(result[_VCLIMAX].iloc[-1], 1.0)
        self.assertEqual(pandas_ta.vclimax(**_climax_frame(bars), drop=10)["VCLIMAX_50_20_2.0_3_10.0_5_0.6"].iloc[-1], 0.0)

    def test_vclimax_warm_up_and_flat_bar(self):
        frame = _climax_frame(_SELLING)
        frame["high"].iloc[10] = frame["low"].iloc[10] = frame["close"].iloc[10] = 100.0
        result = pandas_ta.vclimax(**frame)
        self.assertTrue(result["VCLIMAXr_50"].iloc[:50].isna().all())
        self.assertTrue(result[_VCLIMAX].iloc[:50].isna().all())
        self.assertTrue(result[_VCLIMAX_FT].iloc[:51].isna().all())
        self.assertFalse(result[_VCLIMAX].iloc[50:].isna().any())
        self.assertEqual(result["VCLIMAXp"].iloc[10], 0.5)

    def test_vclimax_no_lookahead(self):
        frame = _climax_frame(_SELLING)
        full = pandas_ta.vclimax(**frame)
        for cut in (54, 58, 59):
            part = pandas_ta.vclimax(**{key: value.iloc[:cut] for key, value in frame.items()})
            np.testing.assert_array_equal(part.to_numpy(), full.iloc[:cut].to_numpy())

    def test_vclimax_close_pct_bounds(self):
        frame = _climax_frame(_SELLING)
        for bad in (0.5, 1, 1.2):
            with self.assertRaisesRegex(ValueError, rf"vclimax\(\) close_pct must be a number > 0.5 and < 1, got {bad}"):
                pandas_ta.vclimax(**frame, close_pct=bad)
