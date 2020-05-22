import numpy as np

def print_latency_stats(data, ident, epoch=0):
  npdata = np.array(data)
  tput = 0

  if epoch > 0:
      tput = len(data) / epoch

  mean = np.mean(npdata)
  median = np.percentile(npdata, 50)
  p75 = np.percentile(npdata, 75)
  p95 = np.percentile(npdata, 95)
  p99 = np.percentile(npdata, 99)
  mx = np.max(npdata)

  p25 = np.percentile(npdata, 25)
  p05 = np.percentile(npdata, 5)
  p01 = np.percentile(npdata, 1)
  mn = np.min(npdata)
  output = ('%s LATENCY:\n\tsample size: %d\n' +
        '\tTHROUGHPUT: %.4f\n'
        '\tmean: %.6f, median: %.6f\n' +
        '\tmin/max: (%.6f, %.6f)\n' +
        '\tp25/p75: (%.6f, %.6f)\n' +
        '\tp5/p95: (%.6f, %.6f)\n' +
        '\tp1/p99: (%.6f, %.6f)') % (ident, len(data), tput, mean,
                                     median, mn, mx, p25, p75, p05, p95,
                                     p01, p99)

  print(output)