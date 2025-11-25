import csv
import matplotlib.pyplot as plt

sizes = []
h2d_page = []
d2h_page = []
h2d_pin = []
d2h_pin = []

with open("results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sizes.append(int(row["size_bytes"]) / (1024 * 1024))  # convert to MB
        h2d_page.append(float(row["pageable_h2d_gb_s"]))
        d2h_page.append(float(row["pageable_d2h_gb_s"]))
        h2d_pin.append(float(row["pinned_h2d_gb_s"]))
        d2h_pin.append(float(row["pinned_d2h_gb_s"]))

plt.plot(sizes, h2d_page, label="Pageable H2D")
plt.plot(sizes, d2h_page, label="Pageable D2H")
plt.plot(sizes, h2d_pin, label="Pinned H2D")
plt.plot(sizes, d2h_pin, label="Pinned D2H")

plt.xlabel("Transfer Size (MB)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("GPU Memory Transfer Bandwidth")
plt.legend()
plt.grid(True)

plt.savefig("bandwidth_plot.png")
plt.show()

