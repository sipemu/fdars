"""Tests for depth measures."""

import numpy as np
import pytest


class TestDepthFunctions:
    """Tests for depth computation."""

    def test_depth_fm(self, sample_fdata_1d):
        from fdapy import depth_fm

        depths = depth_fm(sample_fdata_1d)
        assert depths.shape == (sample_fdata_1d.n_samples,)
        assert np.all(depths >= 0)
        assert np.all(depths <= 1)

    def test_depth_mode(self, sample_fdata_1d):
        from fdapy import depth_mode

        depths = depth_mode(sample_fdata_1d, h=0.5)
        assert depths.shape == (sample_fdata_1d.n_samples,)
        assert np.all(depths >= 0)

    def test_depth_rp(self, sample_fdata_1d):
        from fdapy import depth_rp

        depths = depth_rp(sample_fdata_1d, n_projections=20)
        assert depths.shape == (sample_fdata_1d.n_samples,)
        assert np.all(depths >= 0)

    def test_depth_bd(self, sample_fdata_1d):
        from fdapy import depth_bd

        depths = depth_bd(sample_fdata_1d)
        assert depths.shape == (sample_fdata_1d.n_samples,)
        assert np.all(depths >= 0)
        assert np.all(depths <= 1)

    def test_depth_mbd(self, sample_fdata_1d):
        from fdapy import depth_mbd

        depths = depth_mbd(sample_fdata_1d)
        assert depths.shape == (sample_fdata_1d.n_samples,)
        assert np.all(depths >= 0)
        assert np.all(depths <= 1)

    def test_depth_dispatcher(self, sample_fdata_1d):
        from fdapy import depth

        for method in ["FM", "mode", "RP", "RT", "BD", "MBD", "MEI"]:
            depths = depth(sample_fdata_1d, method=method)
            assert depths.shape == (sample_fdata_1d.n_samples,)

    def test_depth_with_reference(self, sample_fdata_1d):
        from fdapy import depth_fm

        # Use subset as reference
        ref = sample_fdata_1d[0:10]
        depths = depth_fm(sample_fdata_1d, fdataori=ref)
        assert depths.shape == (sample_fdata_1d.n_samples,)
