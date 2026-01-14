import numpy as np

class Utils:

    @staticmethod
    def euclidean(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    @staticmethod
    def hitung_sse_dan_tabel(X, centroids, label):
        print("\n" + "-" * 95)
        print(f"{label:^95}")
        print("-" * 95)
        print(f"{'Data':<6}{'C1':>12}{'C2':>12}{'C3':>12}"
              f"{'Cluster':>12}{'Jarak^2':>12}")
        print("-" * 95)

        sse = 0
        for i, x in enumerate(X):
            jarak = [round(Utils.euclidean(x, c), 4) for c in centroids]
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
