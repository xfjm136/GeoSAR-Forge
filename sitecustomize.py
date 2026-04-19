"""
Runtime compatibility patches for third-party tools launched from this repo.

Currently fixes an ISCE2 + NumPy compatibility bug in
`isceobj.Util.Poly2D.Poly2D.polyfit`, where `np.linalg.lstsq` returns the
residual as a 1-element array and legacy code tries to format it as a scalar.
"""

from __future__ import annotations


def _patch_isce_poly2d() -> None:
    try:
        import numpy as np
    except Exception:
        return

    def _patched_polyfit(self, xin, yin, zin, sig=None, snr=None, cond=None, maxOrder=True):
        x = np.array(xin)
        xmin = np.min(x)
        xnorm = np.max(x) - xmin
        if xnorm == 0:
            xnorm = 1.0
        x = (x - xmin) / xnorm

        y = np.array(yin)
        ymin = np.min(y)
        ynorm = np.max(y) - ymin
        if ynorm == 0:
            ynorm = 1.0
        y = (y - ymin) / ynorm

        z = np.array(zin)
        bigOrder = max(self._azimuthOrder, self._rangeOrder)

        arrList = []
        for ii in range(self._azimuthOrder + 1):
            yfact = np.power(y, ii)
            for jj in range(self._rangeOrder + 1):
                xfact = np.power(x, jj) * yfact
                if maxOrder:
                    if (ii + jj) <= bigOrder:
                        arrList.append(xfact.reshape((x.size, 1)))
                else:
                    arrList.append(xfact.reshape((x.size, 1)))

        A = np.hstack(arrList)

        if sig is not None and snr is not None:
            raise Exception("Only one of sig / snr can be provided")

        if sig is not None:
            snr = 1.0 + 1.0 / sig

        if snr is not None:
            A = A / snr[:, None]
            z = z / snr

        returnVal = True

        val, res, rank, eigs = np.linalg.lstsq(A, z, rcond=cond)
        if np.size(res) > 0:
            # Newer NumPy may return residual as a 1-element ndarray instead of
            # a scalar; reduce it explicitly before formatting/printing.
            chi_sq = float(np.sqrt(float(np.sum(np.asarray(res, dtype=np.float64))) / (1.0 * len(z))))
            print("Chi squared: %f" % chi_sq)
        else:
            print("No chi squared value....")
            print("Try reducing rank of polynomial.")
            returnVal = False

        self.setMeanRange(xmin)
        self.setMeanAzimuth(ymin)
        self.setNormRange(xnorm)
        self.setNormAzimuth(ynorm)

        coeffs = []
        count = 0
        for ii in range(self._azimuthOrder + 1):
            row = []
            for jj in range(self._rangeOrder + 1):
                if maxOrder:
                    if (ii + jj) <= bigOrder:
                        row.append(val[count])
                        count += 1
                    else:
                        row.append(0.0)
                else:
                    row.append(val[count])
                    count += 1
            coeffs.append(row)

        self.setCoeffs(coeffs)
        return returnVal

    _patched_polyfit._insarforge_patched = True  # type: ignore[attr-defined]

    patched_any = False
    for mod_name in (
        "isce.components.isceobj.Util.Poly2D",
        "isceobj.Util.Poly2D",
    ):
        try:
            mod = __import__(mod_name, fromlist=["Poly2D"])
            cls = getattr(mod, "Poly2D", None)
            if cls is None:
                continue
            if getattr(cls.polyfit, "_insarforge_patched", False):
                patched_any = True
                continue
            cls.polyfit = _patched_polyfit
            patched_any = True
        except Exception:
            continue

    return patched_any


_patch_isce_poly2d()
