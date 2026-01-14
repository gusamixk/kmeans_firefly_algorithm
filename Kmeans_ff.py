import numpy as np
np.set_printoptions(precision=3, suppress=True)
np.random.seed(1)

# =================================================
# CLASS FIREFLY CLUSTERING
# =================================================
class FireflyClustering:

    # -------------------------------------------------
    # INISIALISASI
    # -------------------------------------------------
    def __init__(self, X, firefly1, firefly2, alpha=0.2, beta=0.5):
        self.X = X
        self.firefly1 = firefly1
        self.firefly2 = firefly2
        self.alpha = alpha
        self.beta = beta

    # -------------------------------------------------
    # DISTANCE
    # -------------------------------------------------
    def euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # -------------------------------------------------
    # HITUNG SSE + TAMPILKAN TABEL
    # -------------------------------------------------
    def hitung_sse(self, centroids, label):
        print("\n" + "-" * 95)
        print(f"{label:^95}")
        print("-" * 95)
        print(f"{'Data':<6}{'C1':>12}{'C2':>12}{'C3':>12}"
              f"{'Cluster':>12}{'Jarak^2':>12}")
        print("-" * 95)

        sse = 0
        for i, x in enumerate(self.X):
            jarak = [round(self.euclidean(x, c), 4) for c in centroids]
            idx = np.argmin(jarak)
            jarak2 = round(jarak[idx] ** 2, 4)
            sse += jarak2

            print(f"D{i+1:<5}"
                  f"{jarak[0]:>12.4f}"
                  f"{jarak[1]:>12.4f}"
                  f"{jarak[2]:>12.4f}"
                  f"{idx+1:>12}"
                  f"{jarak2:>12.4f}")

        print("-" * 95)
        print(f"{'TOTAL SSE':>70} = {sse:.3f}")
        print("-" * 95)

        return sse

    # -------------------------------------------------
    # ITERASI 1
    # -------------------------------------------------
    def iterasi_1(self):
        print("\n" + "=" * 95)
        print(f"{'ITERASI 1 (FIREFLY AWAL)':^95}")
        print("=" * 95)

        print("\nCentroid Firefly 1:")
        print(self.firefly1)
        sse1 = self.hitung_sse(self.firefly1, "HASIL FIREFLY 1 - ITERASI 1")

        print("\nCentroid Firefly 2:")
        print(self.firefly2)
        sse2 = self.hitung_sse(self.firefly2, "HASIL FIREFLY 2 - ITERASI 1")

        return sse1, sse2

    # -------------------------------------------------
    # UPDATE FIREFLY (REDUP → TERANG)
    # -------------------------------------------------
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

        dim_new = np.round(dim_new, 4)

        if dim_id == 1:
            self.firefly1 = dim_new
        else:
            self.firefly2 = dim_new

    # -------------------------------------------------
    # ITERASI 2
    # -------------------------------------------------
    def iterasi_2(self):
        print("\n" + "=" * 95)
        print(f"{'ITERASI 2 (SETELAH UPDATE)':^95}")
        print("=" * 95)

        print("\nCentroid Firefly 1:")
        print(self.firefly1)
        sse1 = self.hitung_sse(self.firefly1, "HASIL FIREFLY 1 - ITERASI 2")

        print("\nCentroid Firefly 2:")
        print(self.firefly2)
        sse2 = self.hitung_sse(self.firefly2, "HASIL FIREFLY 2 - ITERASI 2")

        return sse1, sse2

    # -------------------------------------------------
    # KESIMPULAN
    # -------------------------------------------------
    def kesimpulan(self, sse1, sse2):
        print("\n" + "=" * 95)
        print(f"{'KESIMPULAN':^95}")
        print("=" * 95)

        best = "Firefly 1" if sse1 <= sse2 else "Firefly 2"
        best_sse = min(sse1, sse2)

        print(f"""
Berdasarkan iterasi ke-2:
- SSE Firefly 1 = {sse1:.3f}
- SSE Firefly 2 = {sse2:.3f}

Firefly terbaik adalah {best}
karena memiliki SSE paling kecil ({best_sse:.3f}).

Firefly Algorithm memilih solusi terbaik
berdasarkan brightness tertinggi (SSE minimum).
""")

# =================================================
# MAIN PROGRAM
# =================================================
X = np.array([
    [1,2,1,2],[1,1,2,2],[2,2,1,1],[2,1,2,1],
    [8,8,9,8],[9,8,8,9],[8,9,8,8],[9,9,9,8],
    [4,5,4,5],[5,4,5,4],[4,4,5,5],[5,5,4,4]
])

firefly1 = np.array([[1,2,1,2],[8,8,8,8],[4,5,4,5]])
firefly2 = np.array([[2,1,2,1],[9,9,8,8],[5,4,5,4]])

model = FireflyClustering(X, firefly1, firefly2)

sse1_i1, sse2_i1 = model.iterasi_1()
model.update_firefly(sse1_i1, sse2_i1)
sse1_i2, sse2_i2 = model.iterasi_2()
model.kesimpulan(sse1_i2, sse2_i2)
