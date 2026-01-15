import numpy as np
from utils import Utils

class FireflyClustering:

    def __init__(self, X, firefly1, firefly2, alpha=0.2, beta=0.5):
        self.X = X
        self.firefly1 = firefly1
        self.firefly2 = firefly2
        self.alpha = alpha
        self.beta = beta

    def iterasi_1(self):
        print("\n" + "=" * 95)
        print(f"{'ITERASI 1 (FIREFLY AWAL)':^95}")
        print("=" * 95)

        print("\nCentroid    Firefly 1:")
        print(self.firefly1)
        sse1 = Utils.hitung_sse_dan_tabel(
            self.X, self.firefly1, "HASIL FIREFLY 1 - ITERASI 1"
        )

        print("\nCentroid Firefly 2:")
        print(self.firefly2)
        sse2 = Utils.hitung_sse_dan_tabel(
            self.X, self.firefly2, "HASIL FIREFLY 2 - ITERASI 1"
        )

        return sse1, sse2

    def update_firefly(self, sse1, sse2):
        print("\n" + "=" * 95)
        print(f"{'EVALUASI BRIGHTNESS':^95}")
        print("=" * 95)

        if sse1 <= sse2:
            bright, dim = self.firefly1, self.firefly2
            dim_id = 2
            print("Firefly 1 → TERANG")
            print("Firefly 2 → REDUP")
        else:
            bright, dim = self.firefly2, self.firefly1
            dim_id = 1
            print("Firefly 2 → TERANG")
            print("Firefly 1 → REDUP")

        dim_new = dim + self.beta * (bright - dim) \
                  + self.alpha * (np.random.rand(*dim.shape) - 0.5)

        dim_new = np.round(dim_new, 3)

        if dim_id == 1:
            self.firefly1 = dim_new
        else:
            self.firefly2 = dim_new

    def iterasi_2(self):
        print("\n" + "=" * 95)
        print(f"{'ITERASI 2 (SETELAH UPDATE)':^95}")
        print("=" * 95)

        print("\nCentroid Firefly 1:")
        print(self.firefly1)
        sse1 = Utils.hitung_sse_dan_tabel(
            self.X, self.firefly1, "HASIL FIREFLY 1 - ITERASI 2"
        )

        print("\nCentroid Firefly 2:")
        print(self.firefly2)
        sse2 = Utils.hitung_sse_dan_tabel(
            self.X, self.firefly2, "HASIL FIREFLY 2 - ITERASI 2"
        )

        return sse1, sse2

    def kesimpulan(self, sse1, sse2):
        print("\n" + "=" * 95)
        print(f"{'KESIMPULAN':^95}")
        print("=" * 95)

        best = "Firefly 1" if sse1 <= sse2 else "Firefly 2"
        best_sse = min(sse1, sse2)

        print(f"""
Firefly terbaik adalah {best}
karena memiliki SSE paling kecil ({best_sse:.3f}).

Firefly Algorithm mengoptimasi centroid
dengan memilih brightness tertinggi (SSE minimum).
""")
