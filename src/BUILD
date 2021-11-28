cc_library(
    name = "gemm_lab",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = [
        "-O3",
        "-mfma",
        "-Ofast",
        "-fopenmp",
    ],
    linkopts = ["-lgomp"],
    visibility = [
        "//example:__subpackages__",
        "//test:__subpackages__",
    ],
)