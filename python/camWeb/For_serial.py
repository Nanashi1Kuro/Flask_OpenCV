def _map(x, in_min, in_max, out_min, out_max):
    val = int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    return val if val > 0 else 0
